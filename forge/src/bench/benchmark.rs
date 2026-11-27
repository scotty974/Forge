//! Mesure du temps et de la RAM d'un traitement découpé en phases.
//!
//! Pour chaque phase on relève :
//! - la durée,
//! - le pic de tas alloué pendant la phase, au-delà de ce qui était déjà alloué
//!   (nécessite `TrackingAllocator` comme allocateur global),
//! - la mémoire physique du processus (RSS / working set) en fin de phase.
//!
//! Le rapport s'affiche dans la console et s'écrit en JSON, au même format que
//! le benchmark Python, pour pouvoir les comparer.

use std::{
    error::Error,
    fs,
    path::Path,
    time::{Duration, Instant},
};

use crate::bench::allocator::{current_bytes, peak_bytes, reset_peak};

pub struct PhaseResult {
    pub name: String,
    pub duration: Duration,
    /// Pic de tas pendant la phase, en plus de ce qui était alloué avant elle.
    pub peak_heap_bytes: usize,
    /// Mémoire physique du processus à la fin de la phase.
    pub rss_bytes: usize,
}

pub struct Benchmark {
    label: String,
    start: Instant,
    end: Option<Instant>,
    heap_baseline: usize,
    max_heap: usize,
    max_rss: usize,
    phases: Vec<PhaseResult>,
    metrics: Vec<(String, f64)>,
}

/// Mémoire physique du processus, 0 si elle n'est pas disponible.
pub fn rss_bytes() -> usize {
    memory_stats::memory_stats()
        .map(|m| m.physical_mem)
        .unwrap_or(0)
}

impl Benchmark {
    pub fn new(label: &str) -> Self {
        let heap_baseline = current_bytes();
        reset_peak();
        Benchmark {
            label: label.to_string(),
            start: Instant::now(),
            end: None,
            heap_baseline,
            max_heap: heap_baseline,
            max_rss: rss_bytes(),
            phases: Vec::new(),
            metrics: Vec::new(),
        }
    }

    /// Exécute `f` en mesurant son temps et sa mémoire, et renvoie son résultat.
    pub fn phase<T, F: FnOnce() -> T>(&mut self, name: &str, f: F) -> T {
        let heap_before = current_bytes();
        reset_peak();

        let t = Instant::now();
        let out = f();
        let duration = t.elapsed();

        let peak = peak_bytes();
        let rss = rss_bytes();
        self.max_heap = self.max_heap.max(peak);
        self.max_rss = self.max_rss.max(rss);

        self.phases.push(PhaseResult {
            name: name.to_string(),
            duration,
            peak_heap_bytes: peak.saturating_sub(heap_before),
            rss_bytes: rss,
        });
        out
    }

    /// Ajoute une métrique de qualité (R2, MAE...) au rapport.
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.push((name.to_string(), value));
    }

    pub fn phases(&self) -> &[PhaseResult] {
        &self.phases
    }

    /// Arrête le chronomètre global : ce qui suit (affichage, graphiques...)
    /// n'est plus compté dans le temps total.
    pub fn finish(&mut self) {
        if self.end.is_none() {
            self.end = Some(Instant::now());
        }
    }

    /// Temps total : de la création jusqu'à `finish`, ou jusqu'à maintenant.
    pub fn total(&self) -> Duration {
        match self.end {
            Some(end) => end - self.start,
            None => self.start.elapsed(),
        }
    }

    /// Pic de tas sur l'ensemble des phases, au-delà de l'état initial.
    pub fn peak_heap_bytes(&self) -> usize {
        self.max_heap.saturating_sub(self.heap_baseline)
    }

    pub fn peak_rss_bytes(&self) -> usize {
        self.max_rss
    }

    pub fn print_report(&self) {
        let total = self.total();
        println!("\n=== Benchmark : {} ===", self.label);
        println!(
            "{:<16} {:>12} {:>14} {:>12}",
            "phase", "temps (ms)", "pic tas (Ko)", "RSS (Mo)"
        );
        for p in &self.phases {
            println!(
                "{:<16} {:>12.3} {:>14.1} {:>12.2}",
                p.name,
                ms(p.duration),
                p.peak_heap_bytes as f64 / 1024.0,
                p.rss_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        println!("{}", "-".repeat(57));
        println!(
            "{:<16} {:>12.3} {:>14.1} {:>12.2}",
            "TOTAL",
            ms(total),
            self.peak_heap_bytes() as f64 / 1024.0,
            self.max_rss as f64 / (1024.0 * 1024.0)
        );
    }

    /// Écrit le rapport en JSON (même format que le benchmark Python).
    pub fn save_json(&self, path: &str, extra: &[(&str, String)]) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_json(extra))?;
        Ok(())
    }

    /// `extra` : champs texte supplémentaires (modèle, dataset...).
    pub fn to_json(&self, extra: &[(&str, String)]) -> String {
        let mut s = String::from("{\n");
        s += "  \"language\": \"rust\",\n";
        s += &format!("  \"label\": \"{}\",\n", escape(&self.label));
        for (k, v) in extra {
            s += &format!("  \"{}\": \"{}\",\n", escape(k), escape(v));
        }
        s += &format!("  \"total_ms\": {:.6},\n", ms(self.total()));
        s += &format!("  \"peak_heap_bytes\": {},\n", self.peak_heap_bytes());
        s += &format!("  \"peak_rss_bytes\": {},\n", self.max_rss);

        s += "  \"phases\": [\n";
        let phases: Vec<String> = self
            .phases
            .iter()
            .map(|p| {
                format!(
                    "    {{\"name\": \"{}\", \"duration_ms\": {:.6}, \"peak_heap_bytes\": {}, \"rss_bytes\": {}}}",
                    escape(&p.name),
                    ms(p.duration),
                    p.peak_heap_bytes,
                    p.rss_bytes
                )
            })
            .collect();
        s += &phases.join(",\n");
        s += "\n  ],\n";

        s += "  \"metrics\": {";
        let metrics: Vec<String> = self
            .metrics
            .iter()
            .map(|(k, v)| format!("\"{}\": {}", escape(k), json_number(*v)))
            .collect();
        s += &metrics.join(", ");
        s += "}\n}\n";
        s
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn json_number(v: f64) -> String {
    if v.is_finite() { format!("{v}") } else { "null".to_string() }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn phase_renvoie_le_resultat_et_enregistre_la_phase() {
        let mut bench = Benchmark::new("test");

        let v = bench.phase("somme", || (1..=10).sum::<i32>());

        assert_eq!(v, 55);
        assert_eq!(bench.phases().len(), 1);
        assert_eq!(bench.phases()[0].name, "somme");
    }

    #[test]
    fn phases_gardent_leur_ordre_et_mesurent_le_temps() {
        let mut bench = Benchmark::new("test");

        bench.phase("a", || std::thread::sleep(Duration::from_millis(5)));
        bench.phase("b", || ());

        let names: Vec<&str> = bench.phases().iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert!(bench.phases()[0].duration >= Duration::from_millis(5));
        assert!(bench.total() >= bench.phases()[0].duration);
    }

    #[test]
    fn finish_fige_le_temps_total() {
        let mut bench = Benchmark::new("test");
        bench.phase("a", || ());
        bench.finish();
        let t1 = bench.total();

        std::thread::sleep(Duration::from_millis(5));
        bench.finish(); // un second appel ne change rien

        assert_eq!(bench.total(), t1);
    }

    #[test]
    fn phase_propage_les_erreurs() {
        let mut bench = Benchmark::new("test");

        let r: Result<i32, String> = bench.phase("echec", || Err("boom".to_string()));

        assert_eq!(r, Err("boom".to_string()));
        assert_eq!(bench.phases().len(), 1);
    }

    #[test]
    fn json_contient_les_champs_attendus() {
        let mut bench = Benchmark::new("immo \"v1\"");
        bench.phase("fit", || ());
        bench.add_metric("r2_test", 0.5);
        bench.add_metric("nan", f64::NAN);

        let json = bench.to_json(&[("model", "gd".to_string())]);

        assert!(json.contains("\"language\": \"rust\""));
        assert!(json.contains("\"label\": \"immo \\\"v1\\\"\""));
        assert!(json.contains("\"model\": \"gd\""));
        assert!(json.contains("\"name\": \"fit\""));
        assert!(json.contains("\"r2_test\": 0.5"));
        assert!(json.contains("\"nan\": null"));
        assert!(json.contains("\"total_ms\""));
        assert!(json.contains("\"peak_rss_bytes\""));
    }
}
