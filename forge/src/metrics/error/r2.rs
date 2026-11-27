use crate::error::error::ForgeError;


pub fn r2(y: &[f32], y_pred: &[f32]) -> Result<f32, ForgeError> {
    if y.len() != y_pred.len() {
        return Err(ForgeError::LengthMismatch {
            context: "r2",
            left: y.len(),
            right: y_pred.len(),
        });
    }

    if y.is_empty() {
        return Err(ForgeError::EmptyData { context: "r2" });
    }

    let y_mean: f32 = y.iter().sum::<f32>() / y.len() as f32;

    // Somme des carrés des résidus : erreurs du modèle
    let ss_res: f32 = y.iter().zip(y_pred).map(|(yi, pi)| (yi - pi).powi(2)).sum();

    // Somme totale des carrés : dispersion de y autour de sa moyenne
    let ss_tot: f32 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    // Cas d'erreur : tous les y sont identiques, le R² n'est pas défini
    if ss_tot == 0.0 {
        return Err(ForgeError::ConstantTarget { context: "r2" });
    }

    Ok(1.0 - ss_res / ss_tot)
}

#[cfg(test)]
mod test {
    use super::*;

    const EPS: f32 = 1e-5;

    // ---------- Cas qui doivent réussir ----------

    #[test]
    fn r2_prediction_parfaite() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((r2(&y, &y).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn r2_prediction_moyenne() {
        let y = vec![1.0, 2.0, 3.0];
        let pred = vec![2.0, 2.0, 2.0];

        assert!(r2(&y, &pred).unwrap().abs() < EPS);
    }

    #[test]
    fn r2_pire_que_la_moyenne() {
        let y = vec![1.0, 2.0, 3.0];
        let pred = vec![3.0, 2.0, 1.0];

        assert!(r2(&y, &pred).unwrap() < 0.0);
    }

    // ---------- Cas qui doivent renvoyer une erreur ----------

    #[test]
    fn r2_y_identiques() {
        let y = vec![4.0, 4.0, 4.0];
        let pred = vec![1.0, 2.0, 3.0];

        let result = r2(&y, &pred);

        assert!(matches!(result, Err(ForgeError::ConstantTarget { .. })));
    }

    #[test]
    fn r2_longueurs_differentes() {
        let result = r2(&[1.0, 2.0, 3.0], &[1.0, 2.0]);

        assert!(matches!(
            result,
            Err(ForgeError::LengthMismatch { left: 3, right: 2, .. })
        ));
    }

    #[test]
    fn r2_vide() {
        let result = r2(&[], &[]);

        assert!(matches!(result, Err(ForgeError::EmptyData { .. })));
    }
}