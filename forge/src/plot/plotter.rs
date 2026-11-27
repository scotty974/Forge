use std::{fs, path::Path};

use plotters::prelude::*;

use crate::error::error::ForgeError;

const OUT_FILENAME: &str = "../result/loss_graph.png";


pub fn plot_loss(arr: &[f32]) -> Result<(), ForgeError> {
    if arr.is_empty() {
        return Err(ForgeError::EmptyData { context: "plot_loss" });
    }

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
    if !min_loss.is_finite() || !max_loss.is_finite() {
        return Err(ForgeError::Plot(
            "plot_loss : la loss contient des NaN ou aucune valeur positive (learning rate trop grand ?)".to_string(),
        ));
    }

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


const PRED_FILENAME: &str = "../result/predictions_graph.png";

/// Nuage "vraie valeur vs prédiction", avec la diagonale y = x (prédiction parfaite).
/// Plus les points sont proches de la diagonale, meilleur est le modèle.
pub fn plot_predictions(y_true: &[f32], y_pred: &[f32]) -> Result<(), ForgeError> {
    if y_true.is_empty() {
        return Err(ForgeError::EmptyData { context: "plot_predictions" });
    }
    if y_true.len() != y_pred.len() {
        return Err(ForgeError::LengthMismatch {
            context: "plot_predictions",
            left: y_true.len(),
            right: y_pred.len(),
        });
    }

    if let Some(parent) = Path::new(PRED_FILENAME).parent() {
        fs::create_dir_all(parent)?;
    }

    // Même échelle sur les deux axes, pour que la diagonale soit à 45°
    let all = y_true.iter().chain(y_pred.iter()).cloned();
    let min = all.clone().fold(f32::INFINITY, f32::min);
    let max = all.fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() {
        return Err(ForgeError::Plot(
            "plot_predictions : les valeurs contiennent des NaN".to_string(),
        ));
    }
    let margin = (max - min) * 0.05;
    let (lo, hi) = (min - margin, max + margin);

    let root_area = BitMapBackend::new(PRED_FILENAME, (900, 900)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Vraie valeur vs prédiction", ("sans-serif", 40))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(lo..hi, lo..hi)?;

    chart
        .configure_mesh()
        .x_desc("Vraie valeur")
        .y_desc("Prédiction")
        .draw()?;

    // Les points : un par ligne du jeu de test
    chart
        .draw_series(
            y_true
                .iter()
                .zip(y_pred)
                .map(|(t, p)| Circle::new((*t, *p), 4, RGBColor(0, 113, 227).mix(0.5).filled())),
        )?
        .label("Prédictions")
        .legend(|(x, y)| Circle::new((x + 10, y), 4, RGBColor(0, 113, 227).filled()));

    // La droite y = x : là où les points seraient si le modèle était parfait
    chart
        .draw_series(LineSeries::new(
            vec![(lo, lo), (hi, hi)],
            RGBColor(232, 89, 12).stroke_width(2),
        ))?
        .label("Prédiction parfaite (y = x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(232, 89, 12).stroke_width(2)));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root_area.present()?;
    Ok(())
}