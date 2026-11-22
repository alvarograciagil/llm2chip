//! How a checkpoint wants its text framed, and which tokens end its answers.

use serde_json::Value;

const QWEN_STOP: [u32; 2] = [248046, 248044];

/// The system prompt Janus's processor puts before every conversation.
const JANUS_SYSTEM: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.";

pub struct PromptStyle {
    /// The checkpoint's own Jinja template, when it publishes one.
    template: Option<String>,
    bos: String,
    eos: String,
    /// Qwen3.5 is framed here, with its reasoning block closed or left open.
    chatml_thinking: bool,
    /// Janus frames a picture between its own tokens, after a system prompt of its own.
    janus: bool,
    pub stop: Vec<u32>,
    /// The tokens that open and close the model's reasoning, if its tokenizer has them.
    pub think: Option<(u32, u32)>,
}

/// Follows a model's reasoning and closes it once it has taken `budget` tokens, as the
/// chip's firmware does.
pub struct ReasoningBudget {
    think: Option<(u32, u32)>,
    budget: u32,
    inside: bool,
    spent: u32,
}

impl ReasoningBudget {
    pub fn new(think: Option<(u32, u32)>, budget: Option<u32>) -> Self {
        Self { think, budget: budget.unwrap_or(u32::MAX), inside: false, spent: 0 }
    }

    pub fn see(&mut self, token: u32) {
        let Some((open, close)) = self.think else { return };
        if token == open {
            (self.inside, self.spent) = (true, 0);
        } else if token == close {
            self.inside = false;
        } else if self.inside {
            self.spent += 1;
        }
    }

    /// The token generation goes on with in place of the one sampled.
    pub fn steer(&mut self, sampled: u32) -> u32 {
        let token = match self.think {
            Some((_, close)) if self.inside && sampled != close && self.spent >= self.budget => close,
            _ => sampled,
        };
        self.see(token);
        token
    }
}

fn think_tokens(token_id: &impl Fn(&str) -> Option<u32>) -> Option<(u32, u32)> {
    Some((token_id("<think>")?, token_id("</think>")?))
}

fn read_json(file: Option<String>) -> Value {
    file.and_then(|text| serde_json::from_str(&text).ok()).unwrap_or(Value::Null)
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
    /// How a checkpoint of `family` frames its text, from the files of its folder that `read`
    /// gives by name (tokenizer_config.json, chat_template.jinja, generation_config.json and
    /// config.json; any may be missing) and its tokenizer's ids.
    pub fn read(family: &str, read: impl Fn(&str) -> Option<String>, token_id: impl Fn(&str) -> Option<u32>) -> Self {
        if family == "qwen3_5" {
            return Self { template: None, bos: String::new(), eos: String::new(), chatml_thinking: true, janus: false, stop: QWEN_STOP.to_vec(), think: think_tokens(&token_id) };
        }
        let settings = read_json(read("tokenizer_config.json"));
        let template = read("chat_template.jinja").or_else(|| settings["chat_template"].as_str().map(String::from));
        let bos = token_text(&settings["bos_token"]).unwrap_or_default();
        let eos = token_text(&settings["eos_token"]).unwrap_or_default();
        let config = read_json(read("config.json"));
        let mut stop = ids_of(&read_json(read("generation_config.json"))["eos_token_id"]);
        if stop.is_empty() {
            stop = ids_of(&config["eos_token_id"]);
        }
        if stop.is_empty() {
            stop.extend(token_id(&eos));
        }
        let janus = config["model_type"].as_str() == Some("janus");
        Self { template, bos, eos, chatml_thinking: false, janus, stop, think: think_tokens(&token_id) }
    }

    /// The text with an image before the question, which the chip reads as one
    /// `<|image_pad|>` that its tokens replace.
    pub fn render_with_image(&self, prompt: &str, thinking: bool) -> Result<String, String> {
        if self.janus {
            // Janus's own framing; the one placeholder becomes as many vectors as the picture has patches.
            return Ok(format!("{}{JANUS_SYSTEM}\n\n<|User|>: <begin_of_image><image_placeholder><end_of_image>\n{prompt}\n\n<|Assistant|>:", self.bos));
        }
        if !self.chatml_thinking {
            return Err("this model does not read images".into());
        }
        let think = if thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        Ok(format!("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n{think}"))
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
        environment.render_str(template, context).map(|text| (text, false)).map_err(|e| format!("the model's chat template fails: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_goes_before_the_question_as_one_placeholder() {
        let qwen = PromptStyle { template: None, bos: String::new(), eos: String::new(), chatml_thinking: true, janus: false, stop: vec![], think: None };
        assert_eq!(qwen.render_with_image("What is it?", false).unwrap(), "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is it?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(style(None, vec![2]).render_with_image("hola", false).is_err());
    }

    fn style(template: Option<&str>, stop: Vec<u32>) -> PromptStyle {
        PromptStyle { template: template.map(String::from), bos: "<s>".into(), eos: "</s>".into(), chatml_thinking: false, janus: false, stop, think: None }
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
}
