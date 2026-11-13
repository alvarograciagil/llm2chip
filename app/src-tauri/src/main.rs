//! A native window around the local engine, which also serves the interface.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

const ADDRESS: &str = "127.0.0.1:7878";

struct Engine(Mutex<Option<Child>>);

fn wait_for_engine() -> bool {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(20) {
        if TcpStream::connect(ADDRESS).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let resources = app.path().resource_dir()?;
            let engine = format!("engine/server{}", std::env::consts::EXE_SUFFIX);
            let child = Command::new(resources.join(engine)).env("LLM2CHIP_DIST", resources.join("web")).spawn()?;
            app.manage(Engine(Mutex::new(Some(child))));
            if !wait_for_engine() {
                return Err("the local engine did not start".into());
            }
            let url = format!("http://{ADDRESS}/").parse()?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("llm2chip")
                .inner_size(1440.0, 900.0)
                .min_inner_size(960.0, 600.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("the application could not be created")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Engine>().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
