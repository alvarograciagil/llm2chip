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
    /// `<|image_pad|>` that its tokens replace, after the system prompt when there is one
    /// (Janus's own when there is not).
    pub fn render_with_image(&self, system: &str, prompt: &str, thinking: bool) -> Result<String, String> {
        if self.janus {
            // Janus's own framing; the one placeholder becomes as many vectors as the picture has patches.
            return Ok(format!("{}<|User|>: <begin_of_image><image_placeholder><end_of_image>\n{prompt}\n\n<|Assistant|>:", self.image_system(system)?));
        }
        if !self.chatml_thinking {
            return Err("this model does not read images".into());
        }
        let think = if thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        Ok(format!("{}<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n{think}", self.image_system(system)?))
    }

    /// The framing a question with an image starts with: its system prompt.
    fn image_system(&self, system: &str) -> Result<String, String> {
        if self.janus {
            let system = if system.is_empty() { JANUS_SYSTEM } else { system };
            return Ok(format!("{}{system}\n\n", self.bos));
        }
        if system.is_empty() {
            return Ok(String::new());
        }
        Ok(self.render_conversation(&[("system", system)], false, false)?.0)
    }

    /// One question after a fixed system prompt (none when it is empty), framed for the
    /// model's answer: the text to tokenize, and whether the tokenizer must add its special
    /// tokens.
    pub fn render_with_system(&self, system: &str, prompt: &str, thinking: bool) -> Result<(String, bool), String> {
        if system.is_empty() {
            return self.render(prompt, thinking);
        }
        self.render_conversation(&[("system", system), ("user", prompt)], thinking, true)
    }

    /// Splits `framed`, a conversation that starts with the system prompt `system` (as
    /// `render_with_system` or `render_with_image` frame it), into the system prompt's
    /// tokens, which a chip carries and runs before anything else, and the text that
    /// follows, which the host sends. `encode` tokenizes a text, adding the special tokens
    /// when asked; they go with the system prompt, at the start. Refuses a template that does
    /// not put the system prompt ahead of the rest, or a split whose tokens differ from the
    /// whole conversation's.
    pub fn split_system(&self, system: &str, framed: &str, image: bool, add_special: bool, encode: impl Fn(&str, bool) -> Result<Vec<u32>, String>) -> Result<(Vec<u32>, String), String> {
        let prefix = if image { self.image_system(system)? } else { self.render_conversation(&[("system", system)], false, false)?.0 };
        let Some(rest) = framed.strip_prefix(prefix.as_str()).filter(|_| prefix.contains(system)) else {
            return Err("this model's chat template does not put the system prompt apart, ahead of the question".into());
        };
        let tokens = encode(&prefix, add_special)?;
        let whole = encode(framed, add_special)?;
        if whole.len() < tokens.len() || whole[..tokens.len()] != tokens[..] || encode(rest, false)? != whole[tokens.len()..] {
            return Err("the system prompt's last tokens join the question's first, so the chip cannot carry it apart: end it with a full stop or a new line".into());
        }
        Ok((tokens, rest.to_string()))
    }

    /// The text to tokenize, and whether the tokenizer must add its special tokens.
    pub fn render(&self, prompt: &str, thinking: bool) -> Result<(String, bool), String> {
        self.render_conversation(&[("user", prompt)], thinking, true)
    }

    /// A conversation, `(role, text)` in order, framed for the model's answer when `answer`
    /// (the user's turn last), or as it stands; the same as `render` for one question.
    pub fn render_conversation(&self, messages: &[(&str, &str)], thinking: bool, answer: bool) -> Result<(String, bool), String> {
        if self.chatml_thinking {
            let think = if thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" };
            let mut text = String::new();
            for (role, content) in messages {
                // An earlier answer keeps the closed reasoning it was written after, so the
                // cache a conversation keeps still matches the text.
                let lead = if *role == "assistant" { "<think>\n\n</think>\n\n" } else { "" };
                text.push_str(&format!("<|im_start|>{role}\n{lead}{content}<|im_end|>\n"));
            }
            if answer {
                text.push_str(&format!("<|im_start|>assistant\n{think}"));
            }
            return Ok((text, false));
        }
        let Some(template) = &self.template else {
            let joined: Vec<&str> = messages.iter().map(|(_, c)| *c).collect();
            return Ok((joined.join("\n"), true));
        };
        let mut environment = minijinja::Environment::new();
        environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function("raise_exception", |message: String| -> Result<String, minijinja::Error> { Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message)) });
        environment.add_function("strftime_now", |_format: String| String::new());
        let context = minijinja::context! {
            messages => messages.iter().map(|(role, content)| minijinja::context! { role => *role, content => *content }).collect::<Vec<_>>(),
            add_generation_prompt => answer,
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
        assert_eq!(qwen.render_with_image("", "What is it?", false).unwrap(), "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is it?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(style(None, vec![2]).render_with_image("", "hola", false).is_err());
    }

    /// A tokenizer for tests: a token per character, and 1 before them with the special ones.
    fn by_character(text: &str, special: bool) -> Result<Vec<u32>, String> {
        Ok(special.then_some(1).into_iter().chain(text.chars().map(|c| c as u32)).collect())
    }

    #[test]
    fn a_system_prompt_goes_first_and_the_chip_carries_its_tokens() {
        let qwen = PromptStyle { template: None, bos: String::new(), eos: String::new(), chatml_thinking: true, janus: false, stop: vec![], think: None };
        let (framed, special) = qwen.render_with_system("Answer briefly.", "Hi", false).unwrap();
        assert_eq!(framed, "<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let (tokens, rest) = qwen.split_system("Answer briefly.", &framed, false, special, by_character).unwrap();
        assert_eq!(tokens, by_character("<|im_start|>system\nAnswer briefly.<|im_end|>\n", false).unwrap());
        assert!(rest.starts_with("<|im_start|>user\nHi"));
        // With an image the system prompt goes before it too.
        let image = qwen.render_with_image("Answer briefly.", "What is it?", false).unwrap();
        assert!(image.starts_with("<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\n<|vision_start|>"));
        assert!(qwen.split_system("Answer briefly.", &image, true, false, by_character).is_ok());
        // Without one, nothing changes.
        assert_eq!(qwen.render_with_system("", "Hi", false).unwrap(), qwen.render("Hi", false).unwrap());
    }

    #[test]
    fn the_special_tokens_go_with_the_system_prompt() {
        let plain = style(None, vec![2]);
        let (framed, special) = plain.render_with_system("Be kind.", "hola", false).unwrap();
        assert!(special);
        let (tokens, rest) = plain.split_system("Be kind.", &framed, false, special, by_character).unwrap();
        assert_eq!(tokens[0], 1);
        assert_eq!(rest, "\nhola");
    }

    #[test]
    fn a_template_that_mixes_the_system_prompt_into_the_question_is_refused() {
        let llama2 = "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}{% set s = m['content'] %}{% else %}[INST] {{ s }} {{ m['content'] }} [/INST]{% endif %}{% endfor %}";
        let mixed = style(Some(llama2), vec![2]);
        let (framed, special) = mixed.render_with_system("Be kind.", "hola", false).unwrap();
        assert!(mixed.split_system("Be kind.", &framed, false, special, by_character).is_err());
    }

    #[test]
    fn janus_keeps_its_own_system_prompt_unless_the_chip_has_one() {
        let janus = PromptStyle { template: None, bos: "<s>".into(), eos: String::new(), chatml_thinking: false, janus: true, stop: vec![], think: None };
        assert!(janus.render_with_image("", "What?", false).unwrap().starts_with(&format!("<s>{JANUS_SYSTEM}\n\n<|User|>")));
        assert!(janus.render_with_image("Be brief.", "What?", false).unwrap().starts_with("<s>Be brief.\n\n<|User|>"));
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
