use std::{error::Error, fs::File};

use ml::{
    linear::multiple_linear::MultipleLinearRegression, metrics::error::{mae::mae, r2::r2}, plot::plotter::{plot_loss, plot_predictions}, preprocessing::scaler::StandardScaler, utils::split::train_test_split,
};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("../data/immobilier.csv")?;
    let mut rdr = csv::ReaderBuilder::new().from_reader(file);

    let headers = rdr.headers()?.clone();

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<f32> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let n_cols = record.len();

        let mut row: Vec<f32> = Vec::with_capacity(n_cols - 1);
        for j in 0..n_cols - 1 {
            row.push(record[j].trim().parse()?);
        }
        x.push(row);
        y.push(record[n_cols - 1].trim().parse()?);
    }

    println!("{} lignes, {} variables", x.len(), x[0].len());

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42));

    // Scaler
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train);
    let x_train_scaled = scaler.transform(&x_train);
    let x_test_scaled = scaler.transform(&x_test);

    // Entraînement
    let mut model = MultipleLinearRegression::new(0.005, 100_000).with_tolerance(1e-6);
    model.fit(&x_train_scaled, &y_train);

    // Évaluation sur le train et sur le test
    let pred_train = model.predict(&x_train_scaled);
    let pred = model.predict(&x_test_scaled);

    println!("MSE finale (train) : {:?}", model.loss_history().last());

    // println!("\nPoids (échelle normalisée) :");
    // for (j, w) in model.weights().iter().enumerate() {
    //     println!("  {:<40} {:>8.3}", &headers[j], w);
    // }
    // println!("  {:<40} {:>8.3}", "biais", model.bias());

    println!("\nR2  train : {:.4}", r2(&y_train, &pred_train));
    println!("R2  test  : {:.4}", r2(&y_test, &pred));
    println!("MAE test  : {:.4} points", mae(&y_test, &pred));

    plot_loss(model.loss_history())?;
    println!("\nCourbe de loss : result/loss_graph.png");
    plot_predictions(&y_test, &pred)?;
    println!("\nCourbe de prediction : result/predictions_graph.png");

    Ok(())
}