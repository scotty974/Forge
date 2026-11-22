// src/utils/plot.rs
// Dependance : cargo add plotters@0.3.7
// A declarer dans src/utils/mod.rs : pub mod plot;

use plotters::prelude::*;
use std::error::Error;

const BLEU: RGBColor = RGBColor(0, 113, 227);
const ORANGE: RGBColor = RGBColor(232, 89, 12);

/// Dessine le nuage de points (x, y) et la droite y = m*x + b dans un PNG.
/// Au-dela de `max_points`, on ne dessine qu'un point sur `step` pour garder
/// une image lisible et un temps de rendu raisonnable.
pub fn plot_regression(
    x: &[f32],
    y: &[f32],
    m: f32,
    b: f32,
    path: &str,
    max_points: usize,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(x.len(), y.len(), "x et y doivent avoir la meme longueur");
    assert!(!x.is_empty(), "x ne doit pas etre vide");

    let step = (x.len() / max_points.max(1)).max(1);

    let x_min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_min = y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Regression lineaire (Rust) : y = {:.2}x + {:.2}", m, b),
            ("sans-serif", 32),
        )
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(90)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    chart
        .draw_series(
            x.iter()
                .zip(y)
                .step_by(step)
                .map(|(xi, yi)| Circle::new((*xi, *yi), 2, BLEU.mix(0.4).filled())),
        )?
        .label("Donnees")
        .legend(|(px, py)| Circle::new((px + 10, py), 4, BLEU.filled()));

    chart
        .draw_series(LineSeries::new(
            vec![(x_min, m * x_min + b), (x_max, m * x_max + b)],
            ORANGE.stroke_width(3),
        ))?
        .label("Droite de regression")
        .legend(|(px, py)| {
            PathElement::new(vec![(px, py), (px + 20, py)], ORANGE.stroke_width(3))
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}