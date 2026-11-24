
pub fn r2(y: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y.len(),
        y_pred.len(),
        "r2 : y et y_pred doivent avoir la même longueur"
    );
    assert!(!y.is_empty(), "r2 : y ne doit pas être vide");

    let y_mean: f32 = y.iter().sum::<f32>() / y.len() as f32;

    // Somme des carrés des résidus : erreurs du modèle
    let ss_res: f32 = y
        .iter()
        .zip(y_pred)
        .map(|(yi, pi)| (yi - pi).powi(2))
        .sum();

    // Somme totale des carrés : dispersion de y autour de sa moyenne
    let ss_tot: f32 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    assert!(
        ss_tot != 0.0,
        "r2 : variance de y nulle, tous les y sont identiques"
    );

    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod test {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn r2_prediction_parfaite() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((r2(&y, &y) - 1.0).abs() < EPS);
    }

    #[test]
    fn r2_prediction_moyenne() {
        let y = vec![1.0, 2.0, 3.0];
        let pred = vec![2.0, 2.0, 2.0];

        assert!(r2(&y, &pred).abs() < EPS);
    }

    #[test]
    fn r2_pire_que_la_moyenne() {
        let y = vec![1.0, 2.0, 3.0];
        let pred = vec![3.0, 2.0, 1.0];

        assert!(r2(&y, &pred) < 0.0);
    }

    #[test]
    #[should_panic(expected = "variance de y nulle")]
    fn r2_y_identiques() {
        let y = vec![4.0, 4.0, 4.0];
        let pred = vec![1.0, 2.0, 3.0];

        r2(&y, &pred);
    }

    #[test]
    #[should_panic(expected = "même longueur")]
    fn r2_longueurs_differentes() {
        r2(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "y ne doit pas être vide")]
    fn r2_vide() {
        r2(&[], &[]);
    }
}