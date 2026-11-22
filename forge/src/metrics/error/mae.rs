pub fn mae(y: &[f32], y_pred: &[f32]) -> f32 {
    let n = y.len() as f32;
    let mae = y
        .iter()
        .zip(y_pred)
        .map(|(ya, yi)| (ya - yi).abs())
        .sum::<f32>()
        / n;

    mae
}

