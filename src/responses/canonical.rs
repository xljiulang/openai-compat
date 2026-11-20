use serde_json::Value;

pub(crate) fn canonical_json_string(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            let mut output = String::from("{");
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).expect("map keys serialize"));
                output.push(':');
                output.push_str(&canonical_json_string(value));
            }
            output.push('}');
            output
        }
        Value::Array(values) => {
            let mut output = String::from("[");
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&canonical_json_string(value));
            }
            output.push(']');
            output
        }
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("strings serialize"),
    }
}

pub(crate) fn canonicalize_object_arguments(raw: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(raw).ok()?;
    value.as_object()?;
    Some(canonical_json_string(&value))
}
