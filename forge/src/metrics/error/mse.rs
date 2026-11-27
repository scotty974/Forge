use crate::error::error::ForgeError;

pub fn mean_squared_error(y: &[f32], y_pred: &[f32]) -> Result<f32, ForgeError> {
    if y.len() != y_pred.len() {
        return Err(ForgeError::LengthMismatch {
            context: "mean_squared_error",
            left: y.len(),
            right: y_pred.len(),
        });
    }
    if y.is_empty() {
        return Err(ForgeError::EmptyData { context: "mean_squared_error" });
    }

    let n = y.len() as f32;
    let mse = y
        .iter()
        .zip(y_pred)
        .map(|(ya, yi)| (ya - yi).powi(2))
        .sum::<f32>()
        / n;

    Ok(mse)
}
