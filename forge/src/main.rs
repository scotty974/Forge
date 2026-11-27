use std::fs::File;

use ml::{
    error::error::ForgeError,
    linear::multiple_linear::MultipleLinearRegression, metrics::error::{mae::mae, r2::r2}, plot::plotter::{plot_loss, plot_predictions}, preprocessing::scaler::StandardScaler, utils::split::train_test_split,
};

fn main() -> Result<(), ForgeError> {
    let file = File::open("../data/immobilier.csv")?;
    let mut rdr = csv::ReaderBuilder::new().from_reader(file);

    let headers = rdr.headers()?.clone();

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<f32> = Vec::new();

    // Ligne 1 = en-têtes, les données commencent à la ligne 2
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let line = i + 2;
        let n_cols = record.len();

        if n_cols < 2 {
            return Err(ForgeError::LengthMismatch {
                context: "lecture CSV : au moins 2 colonnes attendues",
                left: n_cols,
                right: headers.len(),
            });
        }

        let mut row: Vec<f32> = Vec::with_capacity(n_cols - 1);
        for (column, field) in record.iter().enumerate() {
            let value: f32 = field.trim().parse().map_err(|_| ForgeError::Parse {
                line,
                column,
                value: field.to_string(),
            })?;
            if column == n_cols - 1 {
                y.push(value);
            } else {
                row.push(value);
            }
        }
        x.push(row);
    }

    if x.is_empty() {
        return Err(ForgeError::EmptyData { context: "lecture CSV" });
    }

    println!("{} lignes, {} variables", x.len(), x[0].len());

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42))?;

    // Scaler
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;
    let x_train_scaled = scaler.transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    // Entraînement
    let mut model = MultipleLinearRegression::new(0.005, 100_000).with_tolerance(1e-6);
    model.fit(&x_train_scaled, &y_train)?;

    // Évaluation sur le train et sur le test
    let pred_train = model.predict(&x_train_scaled)?;
    let pred = model.predict(&x_test_scaled)?;

    println!("MSE finale (train) : {:?}", model.loss_history().last());

    // println!("\nPoids (échelle normalisée) :");
    // for (j, w) in model.weights().iter().enumerate() {
    //     println!("  {:<40} {:>8.3}", &headers[j], w);
    // }
    // println!("  {:<40} {:>8.3}", "biais", model.bias());

    println!("\nR2  train : {:.4}", r2(&y_train, &pred_train)?);
    println!("R2  test  : {:.4}", r2(&y_test, &pred)?);
    println!("MAE test  : {:.4} points", mae(&y_test, &pred)?);

    plot_loss(model.loss_history())?;
    println!("\nCourbe de loss : result/loss_graph.png");
    plot_predictions(&y_test, &pred)?;
    println!("\nCourbe de prediction : result/predictions_graph.png");

    Ok(())
}