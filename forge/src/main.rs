use std::{error::Error, fs::File};

use ml::{linear::multiple_linear::MultipleLinearRegression, metrics::error::r2::r2};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("../data/multi_data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x_train: Vec<Vec<f32>> = Vec::new();
    let mut y_train: Vec<f32> = Vec::new();
    let mut x_test: Vec<Vec<f32>> = Vec::new();
    let mut y_test: Vec<f32> = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let n_cols = record.len();

        // Toutes les colonnes sauf la dernière : les variables (x1, x2, x3)
        let mut row: Vec<f32> = Vec::with_capacity(n_cols - 1);
        for j in 0..n_cols - 1 {
            row.push(record[j].parse()?);
        }
        // Dernière colonne : la cible y
        let target: f32 = record[n_cols - 1].parse()?;

        // Une ligne sur cinq en test
        if i % 5 == 0 {
            x_test.push(row);
            y_test.push(target);
        } else {
            x_train.push(row);
            y_train.push(target);
        }
    }

    let mut model = MultipleLinearRegression::new(0.01, 5000);
    model.fit(&x_train, &y_train);
    let pred = model.predict(&x_test);
    let score = r2(&y_test, &pred);

    println!("Poids : {:?}", model.weights());
    println!("Biais : {:?}", model.bias());
    println!("5 premières prédictions : {:?}", &pred[..5]);
    println!("5 vraies valeurs        : {:?}", &y_test[..5]);
    println!("R2 : {:?}", score);

    Ok(())
}