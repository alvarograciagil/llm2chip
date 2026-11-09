//! How a checkpoint wants its text framed, and which tokens end its answers.

use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

const QWEN_STOP: [u32; 2] = [248046, 248044];

pub struct PromptStyle {
    /// The checkpoint's own Jinja template, when it publishes one.
    template: Option<String>,
    bos: String,
    eos: String,
    /// Qwen3.5 is framed here, with its reasoning block closed or left open.
    chatml_thinking: bool,
    pub stop: Vec<u32>,
}

fn read_json(path: &Path) -> Value {
    std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or(Value::Null)
}

fn token_text(value: &Value) -> Option<String> {
    value.as_str().or_else(|| value.get("content")?.as_str()).map(String::from)
}

fn ids_of(value: &Value) -> Vec<u32> {
    match value {
        Value::Number(n) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Value::Array(list) => list.iter().filter_map(Value::as_u64).map(|v| v as u32).collect(),
        _ => Vec::new(),
    }
}

impl PromptStyle {
    pub fn read(dir: &Path, family: &str, tokenizer: &Tokenizer) -> Self {
        if family == "qwen3_5" {
            return Self { template: None, bos: String::new(), eos: String::new(), chatml_thinking: true, stop: QWEN_STOP.to_vec() };
        }
        let settings = read_json(&dir.join("tokenizer_config.json"));
        let template = std::fs::read_to_string(dir.join("chat_template.jinja")).ok().or_else(|| settings["chat_template"].as_str().map(String::from));
        let bos = token_text(&settings["bos_token"]).unwrap_or_default();
        let eos = token_text(&settings["eos_token"]).unwrap_or_default();
        let mut stop = ids_of(&read_json(&dir.join("generation_config.json"))["eos_token_id"]);
        if stop.is_empty() {
            stop = ids_of(&read_json(&dir.join("config.json"))["eos_token_id"]);
        }
        if stop.is_empty() {
            stop.extend(tokenizer.token_to_id(&eos));
        }
        Self { template, bos, eos, chatml_thinking: false, stop }
    }

    /// The text to tokenize, and whether the tokenizer must add its special tokens.
    pub fn render(&self, prompt: &str, thinking: bool) -> Result<(String, bool), String> {
        if self.chatml_thinking {
            let think = if thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" };
            return Ok((format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{think}"), false));
        }
        let Some(template) = &self.template else {
            return Ok((prompt.to_string(), true));
        };
        let mut environment = minijinja::Environment::new();
        environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function("raise_exception", |message: String| -> Result<String, minijinja::Error> { Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message)) });
        environment.add_function("strftime_now", |_format: String| String::new());
        let context = minijinja::context! {
            messages => vec![minijinja::context! { role => "user", content => prompt }],
            add_generation_prompt => true,
            enable_thinking => thinking,
            bos_token => self.bos.as_str(),
            eos_token => self.eos.as_str(),
        };
        environment.render_str(template, context).map(|text| (text, false)).map_err(|e| format!("la plantilla de chat del modelo falla: {e}"))
    }

    /// The chip's firmware compares against two tokens.
    pub fn stop_pair(&self) -> [u32; 2] {
        match self.stop.as_slice() {
            [] => [u32::MAX; 2],
            [one] => [*one; 2],
            [one, two, ..] => [*one, *two],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(template: Option<&str>, stop: Vec<u32>) -> PromptStyle {
        PromptStyle { template: template.map(String::from), bos: "<s>".into(), eos: "</s>".into(), chatml_thinking: false, stop }
    }

    #[test]
    fn a_checkpoints_own_template_frames_the_prompt() {
        let llama2 = "{{ bos_token }}{% for m in messages %}[INST] {{ m['content'].strip() }} [/INST]{% endfor %}";
        assert_eq!(style(Some(llama2), vec![2]).render(" hola ", false).unwrap(), ("<s>[INST] hola [/INST]".to_string(), false));
    }

    #[test]
    fn without_a_template_the_text_goes_as_it_is_with_the_special_tokens() {
        assert_eq!(style(None, vec![2]).render("hola", false).unwrap(), ("hola".to_string(), true));
    }

    #[test]
    fn a_template_that_refuses_reports_it() {
        let error = style(Some("{{ raise_exception('solo sistema') }}"), vec![]).render("hola", false).unwrap_err();
        assert!(error.contains("solo sistema"), "{error}");
    }

    #[test]
    fn stop_tokens_fit_the_chips_two_slots() {
        assert_eq!(style(None, vec![]).stop_pair(), [u32::MAX; 2]);
        assert_eq!(style(None, vec![7]).stop_pair(), [7, 7]);
        assert_eq!(style(None, vec![7, 8, 9]).stop_pair(), [7, 8]);
    }
}
