use std::{error::Error, fs, path::Path};

use plotters::prelude::*;

const OUT_FILENAME: &str = "../result/loss_graph.png";


pub fn plot_loss(arr: &[f32]) -> Result<(), Box<dyn Error>> {
    assert!(!arr.is_empty(), "plot_loss : l'historique de loss est vide");

    // Crée le dossier result/ s'il n'existe pas
    if let Some(parent) = Path::new(OUT_FILENAME).parent() {
        fs::create_dir_all(parent)?;
    }

    // Bornes de l'axe vertical : l'échelle log exige des valeurs strictement positives
    let min_loss = arr
        .iter()
        .cloned()
        .filter(|v| *v > 0.0)
        .fold(f32::INFINITY, f32::min);
    let max_loss = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_loss.is_finite() && max_loss.is_finite(),
        "plot_loss : la loss contient des NaN ou aucune valeur positive (learning rate trop grand ?)"
    );

    let root_area = BitMapBackend::new(OUT_FILENAME, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Courbe de loss (MSE)", ("sans-serif", 40))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(90)
        .build_cartesian_2d(0..arr.len(), (min_loss..max_loss).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("MSE (échelle log)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        arr.iter()
            .enumerate()
            .filter(|(_, loss)| **loss > 0.0)
            .map(|(epoch, loss)| (epoch, *loss)),
        RGBColor(0, 113, 227).stroke_width(2),
    ))?;

    root_area.present()?;
    Ok(())
}