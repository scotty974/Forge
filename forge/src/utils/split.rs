pub fn train_test_split<T: Clone>(x: &[T], y: &[f32], size: f32) -> (Vec<T>, Vec<T>, Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), y.len(), "split : x et y ne sont pas de la même longueur");
    assert!(size > 0.0 &&size < 1.0, "split : Veuillez rentrer une bonne valeur de split");
    let x_size = x.len() as f32;

    let slice = x_size - x_size * size;
    let split_index = slice.round() as usize;

    let (x_train, x_test) = x.split_at(split_index);
    let (y_train, y_test) = y.split_at(split_index);

    (x_train.to_vec(), x_test.to_vec(), y_train.to_vec(), y_test.to_vec())

}

#[cfg(test)]
mod test {
    use super::*;

    /// x = [0, 1, ..., n-1] et y = x × 10 : on voit tout de suite si x et y restent alignés.
    fn donnees(n: usize) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|xi| xi * 10.0).collect();
        (x, y)
    }

    #[test]
    fn split_tailles_80_20() {
        let (x, y) = donnees(10);

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);

        assert_eq!(x_train.len(), 8);
        assert_eq!(x_test.len(), 2);
        assert_eq!(y_train.len(), 8);
        assert_eq!(y_test.len(), 2);
    }

    #[test]
    fn split_prend_les_dernieres_lignes_en_test() {
        let (x, y) = donnees(10);

        let (_, x_test, _, y_test) = train_test_split(&x, &y, 0.2);

        assert_eq!(x_test, vec![8.0, 9.0]);
        assert_eq!(y_test, vec![80.0, 90.0]);
    }

    #[test]
    fn split_garde_x_et_y_alignes() {
        let (x, y) = donnees(50);

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3);

        for (xi, yi) in x_train.iter().zip(&y_train) {
            assert_eq!(*yi, xi * 10.0);
        }
        for (xi, yi) in x_test.iter().zip(&y_test) {
            assert_eq!(*yi, xi * 10.0);
        }
    }

    #[test]
    fn split_ne_perd_ni_ne_duplique_aucune_ligne() {
        let (x, y) = donnees(37);

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25);

        // Train puis test, recollés, doivent redonner exactement l'original
        let x_recolle: Vec<f32> = x_train.iter().chain(&x_test).cloned().collect();
        let y_recolle: Vec<f32> = y_train.iter().chain(&y_test).cloned().collect();
        assert_eq!(x_recolle, x);
        assert_eq!(y_recolle, y);
    }

    #[test]
    fn split_arrondit_au_plus_proche() {
        // 7 lignes, test_size = 0.2 : train = 7 - 1.4 = 5.6, arrondi à 6
        let (x, y) = donnees(7);

        let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.2);

        assert_eq!(x_train.len(), 6);
        assert_eq!(x_test.len(), 1);
    }

    #[test]
    #[should_panic(expected = "bonne valeur de split")]
    fn split_size_superieur_a_1() {
        let (x, y) = donnees(10);
        train_test_split(&x, &y, 1.5);
    }

    #[test]
    #[should_panic(expected = "bonne valeur de split")]
    fn split_size_zero() {
        let (x, y) = donnees(10);
        train_test_split(&x, &y, 0.0);
    }

    #[test]
    #[should_panic(expected = "bonne valeur de split")]
    fn split_size_negatif() {
        let (x, y) = donnees(10);
        train_test_split(&x, &y, -0.2);
    }

    #[test]
    #[should_panic(expected = "pas de la même longueur")]
    fn split_longueurs_differentes() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        train_test_split(&x, &y, 0.2);
    }
}