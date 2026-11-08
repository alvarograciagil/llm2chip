use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Maturity {
    Probada,
    Plausible,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityRisk {
    Ninguno,
    Medir,
    Alto,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct OptInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub maturity: Maturity,
    pub quality_risk: QualityRisk,
    pub what: &'static str,
    pub effect: &'static str,
}

const fn opt(
    id: &'static str,
    name: &'static str,
    maturity: Maturity,
    quality_risk: QualityRisk,
    what: &'static str,
    effect: &'static str,
) -> OptInfo {
    OptInfo { id, name, maturity, quality_risk, what, effect }
}

use Maturity::*;
use QualityRisk::*;

/// Effects are assumptions to be replaced by measurements as the project matures.
const CATALOG: &[OptInfo] = &[
    opt("embedding_externo", "Embedding en DRAM externa", Probada, Ninguno,
        "La tabla de embedding solo se lee una fila por token, así que no necesita estar junto al cálculo.",
        "Saca el embedding de la SRAM y añade un controlador de DRAM."),
    opt("salto_ceros", "Saltar pesos a cero", Probada, Ninguno,
        "No se activa el multiplicador cuando el peso es cero.",
        "−10 % de energía de multiplicación."),
    opt("reparacion_por_compilador", "Reparación por compilador", Probada, Ninguno,
        "El compilador coloca los pesos evitando las celdas defectuosas de cada chip.",
        "Redundancia del 3 % al 1 %."),
    opt("apagado_dies_inactivos", "Apagar dies inactivos", Probada, Ninguno,
        "Con un solo usuario solo trabaja un die cada vez; el resto corta la alimentación de su lógica.",
        "−90 % de fuga de lógica en los dies parados."),
    opt("checkpoint_estado", "Checkpoints de conversación", Probada, Ninguno,
        "Guarda el estado en DRAM barata para no releer el prefijo común de cada petición.",
        "Añade un controlador de DRAM; elimina la relectura del prompt."),
    opt("sin_ecc_con_refresco", "Sin ECC, reescritura periódica", Plausible, Medir,
        "Un bit invertido en un peso apenas se nota. Se reescribe la imagen de vez en cuando.",
        "−11 % de área de SRAM."),
    opt("sin_multiplicadores", "Sin multiplicadores por peso", Plausible, Ninguno,
        "Se precalculan los múltiplos de cada activación y cada peso solo elige cuál sumar.",
        "Multiplicador al 40 % de área, 60 % de energía y mitad de retardo."),
    opt("aritmetica_distribuida", "Aritmética distribuida", Plausible, Ninguno,
        "La activación se procesa bit a bit con sumas precalculadas, como en los filtros FPGA clásicos.",
        "Multiplicador al 15 % de área, ×4 ciclos de pesos, 50 % de energía."),
    opt("compresion_pesos", "Compresión de pesos", Plausible, Ninguno,
        "Los pesos se leen en orden, así que un descompresor en flujo por celda es viable.",
        "Pesos al 75 % (por medir); +8 µm² por celda."),
    opt("sram_baja_tension", "SRAM de pesos a baja tensión", Plausible, Medir,
        "Cada peso se lee una vez por token: se puede usar una tensión menor y bloques lentos.",
        "−30 % de energía de lectura y −50 % de fuga."),
    opt("sram_apilada_3d", "SRAM apilada en 3D", Plausible, Ninguno,
        "Un die de SRAM unido sobre la lógica por hybrid bonding, como en las cachés apiladas.",
        "Huella de la SRAM de pesos al 50 %. Coste y calor mayores."),
    opt("poda_pesos", "Poda de pesos pequeños", Plausible, Medir,
        "Se eliminan los pesos de menor magnitud y se almacena solo lo que queda.",
        "Pesos al 85 %. Requiere revalidar la calidad."),
    opt("edram_ganancia", "eDRAM de celda de ganancia", Experimental, Medir,
        "Celda de pocos transistores cuya lectura constante podría servir de refresco.",
        "Densidad ×2 en los pesos. Disponibilidad limitada en nodos avanzados."),
    opt("cim_analogico", "Cálculo analógico en SRAM", Experimental, Alto,
        "La suma ponderada se hace en la propia matriz de SRAM mediante corrientes.",
        "Multiplicador al 30 % de área y 20 % de energía; añade ruido a los resultados."),
    opt("baja_rango", "Factorización de bajo rango", Experimental, Alto,
        "Las matrices grandes se aproximan como producto de dos matrices más pequeñas.",
        "Pesos al 60 %. Cambia el modelo y hay que medir la calidad."),
    opt("cabeza_recortada", "Salida de vocabulario recortada", Experimental, Medir,
        "Solo se calculan los logits de los tokens más frecuentes y el resto se resuelve aparte.",
        "lm_head al 30 %. Los tokens raros pierden precisión."),
    opt("atencion_dispersa", "Atención dispersa", Experimental, Medir,
        "Solo se compara la consulta con una fracción de la caché KV elegida por un índice.",
        "Cálculo de atención y lecturas KV al 25 %."),
];

pub fn catalog() -> &'static [OptInfo] {
    CATALOG
}
