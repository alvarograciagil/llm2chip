//! Show-o in the app: draws a picture from a text, or retouches the marked region of one as
//! a text describes. It runs in floating point on the host, apart from the chip, one job at
//! a time, and the browser asks how it goes.

use serde::Deserialize;
use sim::showo::ShowO;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// What to make: a picture of `prompt`, or, with `picture`, that picture retouched.
#[derive(Deserialize)]
pub struct RetouchRequest {
    /// The folder of Show-o.
    pub path: String,
    pub prompt: String,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_guidance")]
    pub guidance: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub picture: Option<Picture>,
}

/// The largest integer a JavaScript number holds exactly.
const SEED_MASK: u64 = (1 << 53) - 1;

fn default_steps() -> usize {
    16
}

fn default_guidance() -> f32 {
    1.75
}

/// A picture as RGB pixels row by row, and the rectangle to change, its corners in pixels:
/// left and top included, right and bottom not.
#[derive(Deserialize)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub region: [usize; 4],
}

/// A job under way: its step, and in the end the picture or why there is none.
#[derive(Default)]
pub struct RetouchJob {
    pub phase: Mutex<String>,
    pub step: Mutex<(usize, usize)>,
    pub cancel: AtomicBool,
    pub outcome: Mutex<Option<Result<serde_json::Value, String>>>,
}

/// Show-o once opened, with the folder it came from: its weights take seconds to read.
pub type Painter = Arc<Mutex<Option<(PathBuf, Arc<ShowO>)>>>;

/// The pixels of the region, one flag each, or why the rectangle does not fit the picture.
pub fn region_mask(picture: &Picture) -> Result<Vec<bool>, String> {
    let [x0, y0, x1, y1] = picture.region;
    if picture.rgb.len() != picture.width * picture.height * 3 {
        return Err(format!("la imagen dice medir {} × {} y trae {} bytes", picture.width, picture.height, picture.rgb.len()));
    }
    if x0 >= x1 || y0 >= y1 || x1 > picture.width || y1 > picture.height {
        return Err("marca una zona dentro de la imagen".into());
    }
    Ok((0..picture.width * picture.height).map(|i| (x0..x1).contains(&(i % picture.width)) && (y0..y1).contains(&(i / picture.width))).collect())
}

pub fn run(request: &RetouchRequest, job: &RetouchJob, painter: &Painter) -> Result<serde_json::Value, String> {
    let set_phase = |phase: &str| *job.phase.lock().unwrap() = phase.to_string();
    let dir = Path::new(&request.path);
    let model = {
        let mut held = painter.lock().unwrap();
        match held.as_ref() {
            Some((path, model)) if path == dir => model.clone(),
            _ => {
                // Only one Show-o in memory: its weights, in floating point, take gigabytes.
                *held = None;
                set_phase("Abriendo Show-o");
                let model = Arc::new(ShowO::open(dir)?.ok_or("esta carpeta no tiene Show-o")?);
                *held = Some((dir.to_path_buf(), model.clone()));
                model
            }
        }
    };
    let tokenizer = Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| format!("falta el tokenizador de Phi-1.5 en «phi»: {e}"))?;
    let text = tokenizer.encode(request.prompt.as_str(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let steps = request.steps.clamp(1, 64);
    let guidance = request.guidance.clamp(0.0, 10.0);
    // Within 2^53, so the browser, whose numbers are doubles, gives the seed back intact.
    let seed = request.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)) & SEED_MASK;
    let started = std::time::Instant::now();
    let progress = |step: usize, steps: usize| {
        if job.cancel.load(Relaxed) {
            return Err("cancelado".to_string());
        }
        *job.step.lock().unwrap() = (step, steps);
        set_phase(if step < steps { "Rellenando los códigos de la imagen" } else { "El decodificador convierte los códigos en píxeles" });
        Ok(())
    };
    let rgb = match &request.picture {
        Some(picture) => {
            let region = region_mask(picture)?;
            set_phase("Codificando la imagen con MAGVIT-v2");
            model.inpaint(&picture.rgb, picture.width, picture.height, &region, &text, steps, guidance, seed, progress)?
        }
        None => model.draw(&text, steps, guidance, seed, progress)?,
    };
    let side = model.config.side();
    Ok(serde_json::json!({
        "side": side,
        "seconds": started.elapsed().as_secs_f64(),
        "seed": seed,
        "data": format!("data:image/bmp;base64,{}", crate::base64(&crate::bmp(&rgb, side))),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_region_is_the_marked_rectangle_and_must_fit() {
        let picture = |region| Picture { width: 4, height: 3, rgb: vec![0; 36], region };
        let mask = region_mask(&picture([1, 1, 3, 2])).unwrap();
        let marked: Vec<usize> = (0..12).filter(|i| mask[*i]).collect();
        assert_eq!(marked, [5, 6]);
        assert!(region_mask(&picture([2, 0, 2, 3])).is_err(), "vacía");
        assert!(region_mask(&picture([0, 0, 5, 3])).is_err(), "se sale");
        assert!(region_mask(&Picture { rgb: vec![0; 3], ..picture([0, 0, 1, 1]) }).is_err(), "faltan píxeles");
    }
}
