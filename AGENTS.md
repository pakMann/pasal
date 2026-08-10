# AGENTS.md — Fork pasal.id: Hybrid Search & Agentic Legal Retrieval

Dokumen ini adalah panduan kerja untuk coding agent (Claude Code atau setara) yang mengerjakan fork dari [`ilhamfp/pasal`](https://github.com/ilhamfp/pasal). Baca dokumen ini **sebelum** menyentuh kode apa pun, dan baca juga `CLAUDE.md` di root repo upstream — itu berisi konvensi arsitektur asli yang harus tetap dihormati.

## 1. Konteks & Tujuan

Pasal.id adalah platform hukum Indonesia open source: 40.000+ regulasi, 937.000+ pasal terstruktur, disimpan di Supabase (PostgreSQL), diakses lewat MCP server (Python/FastMCP) dan REST API, dengan frontend Next.js.

Pencarian saat ini murni **full-text search** (3-layer: identity fast-path → works FTS → content FTS dengan fallback `ILIKE`, memakai stemmer Indonesia). Ini bagus untuk pencarian berbasis kata kunci/nomor pasal, tapi lemah untuk pertanyaan diskusi hukum berbasis kasus konkret berbahasa natural, di mana kata-kata di pertanyaan user tidak persis sama dengan kata-kata di teks pasal.

**Tujuan fork ini**: menambahkan lapisan **semantic/vector search** di atas FTS yang sudah ada, sehingga hasil pencarian menjadi **hybrid** (FTS + vector, digabung lewat rank fusion), tanpa merusak perilaku pencarian yang sudah dipakai konsumen lain dari MCP server dan REST API publik.

Fork ini **bukan** untuk membangun ulang platform. Semua yang sudah ada (correction flywheel, verification agent, amendment tracking, RLS, audit trail) harus tetap jalan seperti semula.

## 2. Non-Negotiables (Batasan Keras)

- **Jangan ubah perilaku tool/endpoint yang sudah ada** (`search_laws`, `get_pasal`, `get_law_status`, `list_laws`, endpoint REST `/api/v1/search`, dst) kecuali secara eksplisit diminta di scope pekerjaan (lihat §4). Konsumen lama harus tetap kompatibel.
- **Jangan pernah `UPDATE` konten pasal secara langsung.** Semua mutasi konten tetap lewat fungsi `apply_revision()` yang sudah ada (append-only revision trail). Ini berlaku juga untuk kolom embedding — treat sebagai derived data, bukan konten legal, tapi tetap butuh audit trail kapan/dengan model apa embedding dibuat.
- **RLS (Row-Level Security) wajib diteruskan** ke tabel/kolom baru. Data legal bersifat publik-baca; jangan buka celah tulis publik.
- **Migration SQL harus bernomor urut**, mengikuti pola 49 migration yang sudah ada di `packages/supabase/migrations`. Jangan menimpa migration lama.
- **Sanitasi input** untuk query pencarian tetap dipertahankan (pola `[^a-zA-Z0-9 ]` stripping yang sudah ada untuk tsquery) — terapkan pola setara untuk parameter query embedding/hybrid.
- Semua fitur baru harus **backward-compatible secara default**: hybrid search adalah *tambahan mode*, bukan pengganti paksa. FTS murni tetap harus bisa dipanggil.

## 3. Arsitektur Saat Ini (ringkas)

```
Supabase (PostgreSQL)
  └─ 40.143 regulasi, 937.155 pasal, 49 migration, FTS + RLS
       ├── MCP Server (apps/mcp-server, Python + FastMCP, deploy Railway)
       │     tools: search_laws, get_pasal, get_law_status, list_laws
       └── Next.js Web App (apps/web, Vercel)
             routes: /search, /jelajahi, /peraturan/[type], /connect, /api

Data pipeline: scripts/ (Python, httpx, PyMuPDF, BeautifulSoup) — scraping + parsing PDF resmi
Correction Agent: scripts/agent/ — Claude vision-based verification, auto-apply ≥85% confidence
```

## 4. Scope Pekerjaan

Kerjakan dalam urutan ini. Jangan lompat ke tahap berikutnya sebelum tahap sebelumnya lulus kriteria "Definition of Done".

### 4.1 Schema: tambah kolom & extension vector

- Tambah extension `pgvector` ke Supabase lewat migration baru bernomor urut di `packages/supabase/migrations`.
- Tambah kolom embedding (tipe `vector(N)`, N sesuai model embedding yang dipilih di §4.2) ke tabel pasal/artikel yang relevan. Cek nama tabel aktual di migration yang ada sebelum menulis DDL baru — jangan asumsi nama kolom.
- Tambah index `ivfflat` atau `hnsw` pada kolom vector sesuai volume data (937K baris → pertimbangkan `hnsw` untuk recall lebih baik pada skala ini).
- Tambah kolom metadata: `embedding_model` (text), `embedding_generated_at` (timestamptz) — untuk audit dan supaya bisa re-embed massal kalau ganti model nanti.
- RLS policy pada kolom baru: publik boleh baca, tidak ada jalur tulis publik.

### 4.2 Pipeline embedding

- Buat skrip baru di `scripts/` (ikuti pola skrip pipeline yang sudah ada), bukan menimpa skrip lama.
- **Chunking**: satu pasal = satu chunk, tapi teks yang di-embed **bukan** teks pasal mentah saja. Format context-enriched, contoh:
  ```
  {nama_UU} ({nomor} Tahun {tahun}) — {judul_UU}
  {nama_bab, jika ada} > {nama_bagian, jika ada}
  Pasal {nomor_pasal}
  {isi_pasal}
  ```
  Ini penting: teks pasal mentah tanpa context sering ambigu secara semantik.
- **Model embedding**: gunakan model multilingual yang mendukung Bahasa Indonesia dengan baik dan idealnya mendukung hybrid dense+sparse dalam satu model (mis. kelas BGE-M3) supaya kompatibel dengan strategi rank fusion di §4.3. Simpan pilihan model sebagai konfigurasi, bukan hardcode, karena kemungkinan besar akan dievaluasi ulang.
- **Batching & resumability**: pipeline harus bisa di-resume kalau terhenti di tengah jalan (937K baris tidak murah untuk di-embed ulang dari nol). Ikuti pola atomic job claiming (`FOR UPDATE SKIP LOCKED`) yang sudah dipakai scraper pipeline untuk mencegah duplikasi kerja.
- Jangan panggil provider embedding eksternal tanpa rate limiting/backoff eksplisit.

### 4.3 Hybrid search di layer database

- Tulis fungsi SQL (atau kombinasi query di application layer MCP server) yang menjalankan FTS dan vector search secara paralel, lalu menggabungkan hasil dengan **Reciprocal Rank Fusion** (atau metode weighted-score lain yang terdokumentasi alasannya).
- Pertahankan 3-layer fallback FTS yang sudah ada sebagai salah satu jalur dalam hybrid ini — jangan dibuang, karena identity fast-path (deteksi regex nomor regulasi) tetap paling presisi untuk kasus tersebut dan tidak perlu vector search sama sekali.
- Sediakan parameter mode: `fts_only`, `vector_only`, `hybrid` (default `hybrid`) — supaya konsumen lama yang mengandalkan perilaku FTS murni tidak terganggu jika mereka eksplisit minta `fts_only`.

### 4.4 Tool MCP baru / perluasan tool

- Tambah tool baru `search_laws_semantic` (atau perluas `search_laws` dengan parameter `mode`, opsi mana yang dipilih perlu didiskusikan dan dicatat alasannya di PR description) yang mengekspos hybrid search ke konsumen MCP.
- Deskripsi tool (docstring yang dibaca model AI konsumen) harus jelas menjelaskan kapan tool ini dipakai vs `search_laws` biasa — konsumen (agent lain) perlu bisa memilih dengan tepat kapan pencarian semantik lebih relevan daripada keyword.
- Pertimbangkan tool tambahan untuk mendukung alur multi-hop yang sudah didukung tool lain (`get_pasal`, `get_law_status`) — pastikan hasil `search_laws_semantic` mengembalikan `law_id`/nomor pasal yang bisa langsung dipakai sebagai input tool lain, konsisten dengan skema output tool-tool lama.

### 4.5 Reranking (opsional, tahap lanjutan)

- Setelah hybrid retrieval stabil, evaluasi apakah cross-encoder reranker di atas top-k hasil gabungan memberi peningkatan kualitas yang signifikan sebelum menambah kompleksitas & latency.
- Jangan implementasikan ini sebelum §4.1–§4.4 punya baseline evaluasi (lihat §6).

## 5. Yang **Bukan** Tanggung Jawab Fork Ini

Poin-poin berikut sengaja **tidak** masuk scope repo pasal.id — itu tanggung jawab aplikasi chatbot konsumen (agent orchestration layer di luar repo ini):

- Query rewriting dari cerita kasus user menjadi query pencarian.
- Conversation memory / resolusi konteks percakapan multi-turn.
- Orkestrasi agentic loop (kapan panggil tool mana, berapa kali).
- Penyusunan jawaban akhir ke user beserta disclaimer hukum.

MCP server hanya bertanggung jawab menyediakan tool retrieval yang akurat dan grounded. Jangan menaruh logic percakapan di sini.

## 6. Evaluasi & Definition of Done

- Buat set query evaluasi (minimal ~30–50 query) yang mencampur: (a) query berbasis nomor pasal eksak, (b) query berbasis kasus konkret berbahasa natural, (c) query campuran.
- Bandingkan hasil `fts_only` vs `hybrid` pada set ini — dokumentasikan di PR (kualitatif dulu jika belum ada anotasi relevansi; kuantitatif dengan metrik seperti recall@k begitu ada label).
- Setiap perubahan schema harus disertai migration yang bisa dijalankan idempotent di environment bersih (`packages/supabase/migrations`).
- Tool baru harus punya contoh pemanggilan di README/dokumentasi MCP server, konsisten dengan format dokumentasi tool lama.
- Tidak ada regresi pada tool/endpoint lama — jalankan test/skrip yang sudah ada (kalau ada test suite) sebelum PR.
- PR description wajib menjelaskan: model embedding yang dipilih dan alasannya, strategi rank fusion yang dipilih dan alasannya, serta trade-off biaya (storage index vector + biaya generate embedding untuk 937K+ baris).

## 7. Konvensi Kode

Ikuti konvensi yang sudah berlaku di repo (lihat `CLAUDE.md` upstream untuk detail lengkap):
- Python untuk MCP server & pipeline, style konsisten dengan `apps/mcp-server` yang ada.
- SQL migration mengikuti gaya penomoran dan struktur file yang sudah dipakai di `packages/supabase/migrations`.
- Tidak ada mutasi konten langsung di luar `apply_revision()`.
- TypeScript/Next.js hanya disentuh kalau ada perubahan UI pencarian di web app — di luar itu, jangan modifikasi `apps/web` untuk pekerjaan hybrid search backend ini.

## 8. Environment Lokal (Supabase via Podman)

Bagian ini instruksi spesifik untuk environment development lokal di mesin ini — bukan bagian dari arsitektur umum di §3, jadi jangan generalisasi asumsi di bawah ini ke environment lain (CI, staging, production).

Supabase lokal berjalan di **Podman** (bukan Docker). Docker tidak terinstall.

- **Shim wajib:** `~/.local/bin/docker` adalah shim ke `/usr/bin/podman` yang me-rewrite template `{{.Label "key"}}` → `{{index .Labels "key"}}` (podman 4.9 tidak support `.Label`). Jangan hapus file ini — Supabase CLI gagal tanpa shim ini.
- **Selalu set** `DOCKER_HOST=unix:///run/user/1000/podman/podman.sock` sebelum menjalankan perintah `supabase` (via `npx supabase ...`).
- Socket podman: `systemctl --user enable --now podman.socket` jika belum aktif.

### 8.1 Endpoint lokal

| Service | URL |
|---------|-----|
| API (Kong) | http://127.0.0.1:54321 |
| Postgres | postgresql://postgres:postgres@127.0.0.1:54322/postgres |
| Studio | http://127.0.0.1:54323 |
| Mailpit | http://127.0.0.1:54324 |

Project id: `pasal` (container prefix `supabase_*_pasal`). Container `supabase_*_dewirika` milik project lain — jangan disentuh.

### 8.2 Perintah umum

```bash
export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock
npx supabase start      # nyalakan stack
npx supabase stop       # matikan
npx supabase status -o env   # lihat keys & URL
npx supabase migration up    # apply migrasi baru
```

Konfigurasi CLI: `supabase/config.toml` (project root). `supabase/migrations` adalah **symlink** ke `packages/supabase/migrations` — jangan diganti folder biasa.

### 8.3 Gotchas Lokal

- **Port 3000 kadang dipakai app lain** (pernah ada Rails "power commerce"). Kalau `EADDRINUSE` / bentrok, jalankan dengan `-- --port 3100`.
- **Migrasi harus unik nomornya.** Ada dua file `030_*.sql` dan dua `039_*.sql` — `supabase migration up` / `db reset` akan gagal dengan `duplicate key schema_migrations_pkey`. Migrasi awal sudah di-apply manual; untuk migrasi **baru** (termasuk migrasi hybrid search di §4.1) cukup pastikan nomor unik. Jika harus reset total, apply manual via psycopg dengan `ON CONFLICT DO NOTHING` pada `schema_migrations`.
- **`CREATE POLICY IF NOT EXISTS` bukan syntax valid Postgres.** Migration 023 sudah diperbaiki pakai DO-block guard `pg_policies`. Jangan pakai syntax itu di migrasi baru — termasuk saat menambah RLS policy untuk kolom embedding di §4.1.
- **Setelah membuat tabel baru via migrasi lokal**, grants untuk `anon`/`authenticated` sudah ditangani `ALTER DEFAULT PRIVILEGES` (SELECT untuk anon/authenticated, ALL untuk service_role). Untuk tabel yang dibuat di luar migrasi, jalankan `GRANT` manual.
- **Extension `ltree`** dibuat manual di schema `public`. Migrasi tidak membuatnya otomatis — terapkan pola yang sama untuk extension `pgvector` di §4.1 (jangan asumsikan `CREATE EXTENSION` di migration akan cukup tanpa verifikasi manual di environment ini).
- Env files lokal sudah terisi keys lokal (bukan placeholder): `.env`, `apps/web/.env.local`, `apps/mcp-server/.env`, `scripts/.env`. Keys lokal adalah default well-known Supabase local — aman, jangan dipakai untuk production.
- Python venv ada di `.venv/` (root). Pakai `.venv/bin/python` untuk scripts dan MCP server — termasuk skrip pipeline embedding baru di §4.2.
- `psql` tidak terinstall — koneksi DB langsung via psycopg dari `.venv`.

## 9. Referensi

- Repo upstream: https://github.com/ilhamfp/pasal
- `CLAUDE.md` (root repo upstream) — 489 baris spesifikasi arsitektur asli, wajib dibaca sebelum mulai.
- Lisensi: AGPL-3.0 — perubahan yang di-deploy sebagai layanan publik wajib membuka source-nya juga.
