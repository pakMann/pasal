"""Evaluation query set for hybrid search (AGENTS.md §6).

Three categories:
  - pasal:      exact-number/identity queries (e.g. "UU 13 tahun 2003 pasal 81")
  - case:       natural-language case/fact-pattern questions
  - mixed:      legal-topic queries that mention concrete terms but expect
                provisions beyond a pure keyword match

Each query is a dict: {"id", "category", "query", "note"}.
`id` lets the compare script track stability across model/runs.
"""

QUERIES: list[dict] = [
    # --- (a) identity / exact-number queries ---
    {"id": "p-01", "category": "pasal",
     "query": "UU 13 tahun 2003 pasal 81",
     "note": "exact identity fast-path: UU 13/2003, Pasal 81 (upah minimum)"},
    {"id": "p-02", "category": "pasal",
     "query": "UU 1 tahun 1974 pasal 7",
     "note": "marriage age, UU 1/1974 Pasal 7"},
    {"id": "p-03", "category": "pasal",
     "query": "PP 35 tahun 2021 pasal 36",
     "note": "PKWT contract content"},
    {"id": "p-04", "category": "pasal",
     "query": "UU 31 tahun 1999 pasal 2 korupsi",
     "note": "anti-corruption definition"},
    {"id": "p-05", "category": "pasal",
     "query": "UU ITE pasal 27 ayat 3",
     "note": "defamation clause"},
    {"id": "p-06", "category": "pasal",
     "query": "pasal 45 KUHP",
     "note": "KUHP article (kejahatan)"},
    {"id": "p-07", "category": "pasal",
     "query": "KUHPerdata pasal 1320",
     "note": "contract validity conditions"},
    {"id": "p-08", "category": "pasal",
     "query": "PP 36 tahun 2021 pasal 57",
     "note": "Pengupahan — upah minimum"},
    {"id": "p-09", "category": "pasal",
     "query": "UUD 1945 pasal 28H",
     "note": "constitutional right to welfare"},
    {"id": "p-10", "category": "pasal",
     "query": "UU 36 tahun 2009 pasal 113",
     "note": "health law — health workers rights"},

    # --- (b) natural-language case questions ---
    {"id": "c-01", "category": "case",
     "query": "saya dipecat tanpa pesangon dan tanpa surat peringatan tertulis",
     "note": "PHK without severance + warning letters"},
    {"id": "c-02", "category": "case",
     "query": "bos menahan ijazah saya setelah saya berhenti bekerja",
     "note": "employer withholding diploma"},
    {"id": "c-03", "category": "case",
     "query": "kontrak kerja saya diperpanjang terus padahal sudah lima tahun",
     "note": "repeated fixed-term contracts"},
    {"id": "c-04", "category": "case",
     "query": "suami memukul istri apakah bisa dituntut pidana",
     "note": "domestic violence criminal liability"},
    {"id": "c-05", "category": "case",
     "query": "tetangga membangun rumah tanpa izin menutup akses jalan saya",
     "note": "unpermitted building blocking access"},
    {"id": "c-06", "category": "case",
     "query": "saya membeli rumah tapi sertifikat tidak kunjung balik nama",
     "note": "property title transfer delay"},
    {"id": "c-07", "category": "case",
     "query": "perusahaan tidak membayar lembur padahal saya kerja lebih dari 8 jam",
     "note": "unpaid overtime"},
    {"id": "c-08", "category": "case",
     "query": "istri saya menuntut cerai dan ingin hak asuh anak",
     "note": "divorce + child custody"},
    {"id": "c-09", "category": "case",
     "query": "tetangga menyalin dan menjual produk saya tanpa izin",
     "note": "copyright/trade secret infringement"},
    {"id": "c-10", "category": "case",
     "query": "saya kecelakaan kerja di pabrik siapa yang bertanggung jawab membayar biaya",
     "note": "work accident compensation"},

    # --- (c) mixed topic queries ---
    {"id": "m-01", "category": "mixed",
     "query": "hak cuti tahunan karyawan berapa hari",
     "note": "annual leave entitlement"},
    {"id": "m-02", "category": "mixed",
     "query": "upah minimum provinsi 2024 berapa",
     "note": "provincial minimum wage"},
    {"id": "m-03", "category": "mixed",
     "query": "masa percobaan kerja berapa lama",
     "note": "probation period"},
    {"id": "m-04", "category": "mixed",
     "query": "batas umur anak boleh bekerja",
     "note": "minimum working age"},
    {"id": "m-05", "category": "mixed",
     "query": "syarat sah perjanjian jual beli tanah",
     "note": "land sale contract requirements"},
    {"id": "m-06", "category": "mixed",
     "query": "gugatan cerai harus diajukan di pengadilan mana",
     "note": "divorce court jurisdiction"},
    {"id": "m-07", "category": "mixed",
     "query": "denda keterlambatan pembayaran tunjangan anak",
     "note": "child support enforcement"},
    {"id": "m-08", "category": "mixed",
     "query": "hak pekerja perempuan cuti melahirkan",
     "note": "maternity leave rights"},
    {"id": "m-09", "category": "mixed",
     "query": "kewajiban pemberi kerja mengikuti BPJS",
     "note": "mandatory BPJS participation"},
    {"id": "m-10", "category": "mixed",
     "query": "persyaratan mendirikan PT perorangan",
     "note": "single-member PT requirements"},
    {"id": "m-11", "category": "mixed",
     "query": "perlindungan data pribadi konsumen",
     "note": "personal data protection"},
    {"id": "m-12", "category": "mixed",
     "query": "sanksi pencemaran lingkungan oleh pabrik",
     "note": "environmental pollution sanctions"},
    {"id": "m-13", "category": "mixed",
     "query": "hak kompensasi karyawan yang mengundurkan diri",
     "note": "resignation compensation"},
    {"id": "m-14", "category": "mixed",
     "query": "larangan memotong gaji karyawan tanpa alasan",
     "note": "illegal wage deduction"},
    {"id": "m-15", "category": "mixed",
     "query": "kewajiban pengembang menyerahkan sertifikat rumah",
     "note": "developer certificate delivery obligation"},
    {"id": "m-16", "category": "mixed",
     "query": "perlindungan konsumen barang cacat",
     "note": "consumer protection defective goods"},
    {"id": "m-17", "category": "mixed",
     "query": "ketentuan penggelapan uang perusahaan",
     "note": "embezzlement"},
    {"id": "m-18", "category": "mixed",
     "query": "hak warga mengajukan permohonan informasi publik",
     "note": "public information request right"},
    {"id": "m-19", "category": "mixed",
     "query": "syarat menjadi saksi dalam pernikahan",
     "note": "marriage witness requirements"},
    {"id": "m-20", "category": "mixed",
     "query": "batas maksimal upah kerja lembur per hari",
     "note": "overtime pay cap"},
]

# Keep test expectations stable: this is the canonical count in §6.
MIN_QUERY_COUNT = 30
assert len(QUERIES) >= MIN_QUERY_COUNT, f"eval set must have >= {MIN_QUERY_COUNT} queries"


def by_category() -> dict[str, list[dict]]:
    cats: dict[str, list[dict]] = {}
    for q in QUERIES:
        cats.setdefault(q["category"], []).append(q)
    return cats
