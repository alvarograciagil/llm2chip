//! Every template the application offers opens without errors, built the way the
//! application builds it: the template's values over those of a new design.

use arch::{estimate, ChipParams, Severity};
use model_ir::ModelGraph;
use std::path::Path;

/// What `default_params` in the browser bindings gives a new design.
fn new_design() -> serde_json::Value {
    serde_json::to_value(ChipParams { target_tok_s: 4000.0, ..Default::default() }).unwrap()
}

#[test]
fn every_template_opens_without_errors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = std::fs::read_to_string(root.join("crates/wasm/src/lib.rs")).unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(root.join("presets")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(bindings.contains(&format!("presets/{name}")), "{name} is not offered by the browser bindings");
        let preset: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let model_id = preset["model"].as_str().unwrap();
        let config = std::fs::read_to_string(root.join(format!("models/{model_id}.json"))).unwrap();
        let model = ModelGraph::from_config_json(model_id, &config).unwrap();
        let mut params = new_design();
        for (key, value) in preset["params"].as_object().unwrap() {
            params[key] = value.clone();
        }
        let p: ChipParams = serde_json::from_value(params).unwrap();
        let report = estimate(&model, &p).unwrap();
        let errors: Vec<_> = report.violations.iter().filter(|v| v.severity == Severity::Error).map(|v| &v.message).collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
        assert!(report.metric("tok_s").unwrap() > 0.0, "{name}");
        checked += 1;
    }
    assert!(checked >= 7, "only {checked} templates");
}
