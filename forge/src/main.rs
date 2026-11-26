use std::{error::Error, fs::File};

use ml::{
    linear::multiple_linear::MultipleLinearRegression, metrics::error::{mae::mae, r2::r2}, preprocessing::scaler::StandardScaler, utils::split::train_test_split,
};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("../data/multi_data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<f32> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let n_cols = record.len();

        let mut row: Vec<f32> = Vec::with_capacity(n_cols - 1);
        for j in 0..n_cols - 1 {
            row.push(record[j].parse()?);
        }
        x.push(row);
        y.push(record[n_cols - 1].parse()?);
    }

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42));
    
    // scaler
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train);
    let x_trained_scaled = scaler.transform(&x_train);
    let x_test_scaled = scaler.transform(&x_test);

    // 3. Entraînement et évaluation : inchangés
    let mut model = MultipleLinearRegression::new(0.01, 5000);
    model.fit(&x_trained_scaled, &y_train);
    let pred = model.predict(&x_test_scaled);
    let score = r2(&y_test, &pred);
    let mae = mae(&y_test, &pred);

    println!("Poids : {:?}", model.weights());
    println!("Biais : {:?}", model.bias());
    println!("5 premières prédictions : {:?}", &pred[..5]);
    println!("5 vraies valeurs        : {:?}", &y_test[..5]);
    println!("R2 : {:?}", score);
    println!("MAE : {:?}", mae);

    Ok(())
}
