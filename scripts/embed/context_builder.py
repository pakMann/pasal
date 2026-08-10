"""Context-enriched chunk text for embedding (AGENTS.md §4.2).

A raw pasal's text is semantically ambiguous without context, so each chunk is
prefixed with the parent regulation's identity and (where present) its
BAB/BAGIAN ancestors:

    {title} ({number} Tahun {year}) — {tentang/judul}
    {BAB I - heading} > {BAGIAN X - heading}
    Pasal {number}
    {isi pasal}
    {isi ayat anak, satu per baris}

Content is fetched in bulk for a batch of node ids (one query per batch).
"""
import logging
from typing import Any

logger = logging.getLogger("pasal.embed.context")


def _norm_heading(node: dict[str, Any]) -> str:
    """Render a structural ancestor like 'BAB I - KETENTUAN UMUM'."""
    node_type = (node.get("node_type") or "").upper()
    number = node.get("number") or ""
    heading = (node.get("heading") or "").strip()
    label = f"{node_type} {number}".strip()
    if heading:
        return f"{label} - {heading}"
    return label


def build_contexts(
    pasal_nodes: list[dict[str, Any]],
    works: dict[int, dict[str, Any]],
    ancestors: dict[int, list[dict[str, Any]]],
    ayat: dict[int, list[dict[str, Any]]],
) -> dict[int, str]:
    """Assemble enriched text for each pasal node, keyed by node id.

    Args:
        pasal_nodes: rows for the pasal nodes being embedded.
        works: work_id -> work row (with `code` from regulation_types).
        ancestors: node_id -> list of ancestor rows ordered root -> nearest.
        ayat: node_id -> child ayat rows already ordered by sort_order.
    """
    out: dict[int, str] = {}
    for node in pasal_nodes:
        node_id = node["id"]
        work = works.get(node.get("work_id"))
        chain = ancestors.get(node_id, [])
        chain = [a for a in chain if a["node_type"] in ("bab", "bagian", "paragraf")]

        lines: list[str] = []

        if work:
            title = work.get("title_id") or ""
            number = work.get("number") or ""
            year = work.get("year")
            tentang = (work.get("tentang") or "").strip()
            header = f"{title} ({number} Tahun {year})".strip()
            if tentang and tentang != title:
                header += f" — {tentang}"
            lines.append(header)

        if chain:
            lines.append(" > ".join(_norm_heading(a) for a in chain))

        pasal_no = node.get("number") or "?"
        lines.append(f"Pasal {pasal_no}")

        body = (node.get("content_text") or "").strip()
        if body:
            lines.append(body)

        for a in ayat.get(node_id, []):
            text = (a.get("content_text") or "").strip()
            if text:
                lines.append(text)

        out[node_id] = "\n".join(lines).strip()
    return out


def fetch_batch_contexts(conn, node_ids: list[int]) -> tuple[list[dict], dict, dict, dict]:
    """Fetch everything needed to build enriched contexts for a batch.

    Returns (pasal_nodes, works_by_id, ancestors_by_node, ayat_by_node).
    """
    if not node_ids:
        return [], {}, {}, {}

    pasal_nodes: list[dict[str, Any]] = []
    works_by_id: dict[int, dict[str, Any]] = {}
    ancestors_by_node: dict[int, list[dict[str, Any]]] = {}
    ayat_by_node: dict[int, list[dict[str, Any]]] = {}

    with conn.cursor() as cur:
        # 1. The pasal rows themselves
        cur.execute(
            """
            SELECT dn.id, dn.work_id, dn.node_type, dn.number, dn.heading,
                   dn.content_text, dn.parent_id
            FROM document_nodes dn
            WHERE dn.id = ANY(%s)
            """,
            (node_ids,),
        )
        cols = [d.name for d in cur.description]
        pasal_nodes = [dict(zip(cols, r)) for r in cur.fetchall()]

        # 2. Ancestor chain (BAB > BAGIAN > ...) via recursive CTE
        cur.execute(
            """
            WITH RECURSIVE chain AS (
                SELECT id, work_id, node_type, number, heading, content_text,
                       parent_id, 0 AS depth
                FROM document_nodes WHERE id = ANY(%s)
                UNION ALL
                SELECT dn.id, dn.work_id, dn.node_type, dn.number, dn.heading,
                       dn.content_text, dn.parent_id, ch.depth + 1
                FROM document_nodes dn
                JOIN chain ch ON dn.id = ch.parent_id
            )
            SELECT id, node_type, number, heading, parent_id, depth
            FROM chain WHERE depth > 0
            ORDER BY depth DESC
            """,
            (node_ids,),
        )
        acols = [d.name for d in cur.description]
        for r in cur.fetchall():
            row = dict(zip(acols, r))
            ancestors_by_node.setdefault(row["id"], []).append(row)

        # 3. Works metadata
        work_ids = {n["work_id"] for n in pasal_nodes if n.get("work_id")}
        if work_ids:
            cur.execute(
                """
                SELECT w.id, w.frbr_uri, w.title_id, w.number, w.year,
                       w.tentang, rt.code AS reg_code
                FROM works w
                JOIN regulation_types rt ON rt.id = w.regulation_type_id
                WHERE w.id = ANY(%s)
                """,
                (list(work_ids),),
            )
            wcols = [d.name for d in cur.description]
            works_by_id = {r[0]: dict(zip(wcols, r)) for r in cur.fetchall()}

        # 4. Ayat children (ordered)
        pasal_ids = [n["id"] for n in pasal_nodes]
        cur.execute(
            """
            SELECT parent_id, number, content_text
            FROM document_nodes
            WHERE parent_id = ANY(%s) AND node_type = 'ayat'
            ORDER BY parent_id, sort_order
            """,
            (pasal_ids,),
        )
        ycols = [d.name for d in cur.description]
        for r in cur.fetchall():
            row = dict(zip(ycols, r))
            ayat_by_node.setdefault(row["parent_id"], []).append(row)

    return pasal_nodes, works_by_id, ancestors_by_node, ayat_by_node
