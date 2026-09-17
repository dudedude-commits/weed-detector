"""
Веб-интерфейс (Streamlit) для Задания 1 хакатона: "Сорняки на фотографиях
с дрона".

Позволяет:
  1) загружать эталонные фото сорняков (с указанием вида и стадии
     вегетации) — пополняя базу эталонов на диске;
  2) загружать фото полей с дрона;
  3) запускать анализ — найти сорняки, определить вид, стадию и
     количество, получить JSON/CSV и фото с отрисованными рамками.

Запуск:  streamlit run app.py   (или ./start.sh)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import streamlit as st

from pipeline.classifier import ReferenceDatabase
from pipeline.pipeline import (
    ImageResult,
    SPECIES_PALETTE_HEX,
    aggregate_species_stage_counts,
    process_image,
    results_to_dataframe,
    results_to_json,
)

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_data"
STAGE_OPTIONS = ["Розетка", "Стеблевание", "Без стадии (напр. злаки)"]
DEFAULT_SPECIES = ["Бодяк полевой", "Вьюнок полевой", "Пырей ползучий"]

st.set_page_config(page_title="Сорняки на фото с дрона", page_icon="🌾", layout="wide")

# ------------------------------------------------------------------------ #
# Стиль — те же токены и категориальная палитра, что и в dataviz-гайдлайне
# проекта (references/palette.md): фиксированный порядок цветов по видам,
# карточки вместо голых виджетов, единая акцентная зелень.
# ------------------------------------------------------------------------ #
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

      .wd-hero {
        background: linear-gradient(120deg, #0ca30c 0%, #1baf7a 55%, #2a78d6 130%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        color: #ffffff;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(11,11,11,0.12);
      }
      .wd-hero h1 { margin: 0 0 .35rem 0; font-size: 1.9rem; font-weight: 800; color: #fff; }
      .wd-hero p { margin: 0; opacity: .95; font-size: 1.02rem; }
      .wd-hero .wd-steps { margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
      .wd-chip {
        background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px; padding: .25rem .75rem; font-size: .85rem; font-weight: 600;
      }

      .wd-section-title { display:flex; align-items:center; gap:.5rem; margin: .2rem 0 .8rem 0; }
      .wd-section-title .wd-num {
        display:inline-flex; align-items:center; justify-content:center;
        width: 28px; height: 28px; border-radius: 50%;
        background:#0ca30c; color:#fff; font-weight:700; font-size:.9rem; flex-shrink:0;
      }
      .wd-section-title h3 { margin:0; font-size: 1.25rem; }

      .wd-badge {
        display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem;
        border-radius:999px; color:#fff; font-weight:600; font-size:.86rem; margin:.15rem .3rem .15rem 0;
      }
      .wd-badge .wd-dot { width:8px; height:8px; border-radius:50%; background:rgba(255,255,255,.85); }
      .wd-stage-chip {
        display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem;
        background:#f2f1ec; color:#52514e; margin:.1rem .25rem .1rem 0; border:1px solid #e1e0d9;
      }

      div[data-testid="stMetric"] {
        background:#f2f1ec; border:1px solid #e1e0d9; border-radius:14px; padding: .9rem 1.1rem;
      }
      .stButton > button {
        border-radius: 10px; font-weight: 600; border: 1px solid rgba(11,11,11,.08);
        transition: transform .05s ease-in-out, box-shadow .15s ease-in-out;
      }
      .stButton > button:hover { box-shadow: 0 6px 14px rgba(11,11,11,.12); transform: translateY(-1px); }
      .stDownloadButton > button { border-radius: 10px; font-weight: 600; }

      [data-testid="stExpander"] { border-radius: 14px; border: 1px solid #e1e0d9; }
      hr { border-color: #e1e0d9; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_title(num: str, title: str) -> None:
    st.markdown(
        f'<div class="wd-section-title"><span class="wd-num">{num}</span><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def species_color_map(species_names: list[str]) -> dict[str, str]:
    """Один и тот же вид -> один и тот же цвет везде: на бейджах, рамках фото
    и графике. Порядок фиксирован (см. SPECIES_PALETTE_HEX), а не подбирается
    на лету — так 9-й вид не ломает уже выученные пользователем цвета."""
    ordered = sorted(species_names)
    return {name: SPECIES_PALETTE_HEX[i % len(SPECIES_PALETTE_HEX)] for i, name in enumerate(ordered)}


def get_ref_db() -> ReferenceDatabase:
    if "ref_db" not in st.session_state:
        st.session_state["ref_db"] = ReferenceDatabase(REFERENCE_DIR)
    return st.session_state["ref_db"]


def read_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


def bgra_to_png_bytes(bgra: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes() if ok else b""


ref_db = get_ref_db()

# ------------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------------ #
st.markdown(
    """
    <div class="wd-hero">
      <h1>🌾 Сорняки на фотографиях с дрона</h1>
      <p>Задание 1 хакатона — находим сорняки на фото полей, определяем вид, стадию вегетации
      и количество, сохраняем результат в JSON/CSV.</p>
      <div class="wd-steps">
        <span class="wd-chip">1 · Эталоны сорняков</span>
        <span class="wd-chip">2 · Фото полей</span>
        <span class="wd-chip">3 · Результаты и экспорт</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------ #
# Шаг 1. Эталонные фото сорняков
# ------------------------------------------------------------------------ #
with st.container(border=True):
    section_title("1", "Эталонные фото сорняков")
    st.write(
        "Загрузите крупноплановые фото сорняков, подписав вид и стадию вегетации — "
        "так же, как организовано в исходном датасете (`Сорняки/Вид/Стадия/*.jpg`). "
        "Чем больше эталонов — тем точнее распознавание."
    )

    col_summary, col_upload = st.columns([1, 1.4], gap="large")

    summary = ref_db.list_summary()
    color_map = species_color_map(list(summary.keys()) or DEFAULT_SPECIES)

    with col_summary:
        st.markdown("**Текущая база эталонов**")
        if not summary:
            st.info("База пуста — загрузите хотя бы несколько фото по каждому виду.")
        else:
            for species, stages in summary.items():
                color = color_map.get(species, "#898781")
                stage_chips = "".join(
                    f'<span class="wd-stage-chip">{s if s != ReferenceDatabase.NO_STAGE else "без стадии"}: {n}</span>'
                    for s, n in stages.items()
                )
                st.markdown(
                    f'<div style="margin-bottom:.5rem;">'
                    f'<span class="wd-badge" style="background:{color};"><span class="wd-dot"></span>{species}</span>'
                    f"<br/>{stage_chips}</div>",
                    unsafe_allow_html=True,
                )
            total = sum(n for stages in summary.values() for n in stages.values())
            st.caption(f"Всего эталонных фото: {total}")

        if st.button("🗑 Очистить базу эталонов"):
            import shutil

            shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
            REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            st.session_state.pop("ref_db", None)
            st.rerun()

    with col_upload:
        known_species = sorted(set(DEFAULT_SPECIES) | set(summary.keys())) if summary else DEFAULT_SPECIES
        species_choice = st.selectbox("Вид сорняка", known_species + ["➕ Новый вид…"])
        if species_choice == "➕ Новый вид…":
            species_name = st.text_input("Название нового вида", "")
        else:
            species_name = species_choice

        stage_choice = st.selectbox("Стадия вегетации этих фото", STAGE_OPTIONS)
        stage_value = None if stage_choice.startswith("Без стадии") else stage_choice

        ref_files = st.file_uploader(
            "Фото сорняка (можно выбрать сразу несколько)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="ref_uploader",
        )

        if st.button("➕ Добавить в базу эталонов", type="primary", disabled=not (ref_files and species_name)):
            for f in ref_files:
                ref_db.add_image(species_name, stage_value, f.name, f.getvalue())
            st.success(f"Добавлено {len(ref_files)} фото в «{species_name}» ({stage_choice}).")
            st.rerun()

# ------------------------------------------------------------------------ #
# Шаг 2. Фото полей с дрона
# ------------------------------------------------------------------------ #
with st.container(border=True):
    section_title("2", "Фото полей с дрона")
    field_files = st.file_uploader(
        "Загрузите одно или несколько фото полей (JPG/PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="field_uploader",
    )

    min_conf = st.slider(
        "Минимальная уверенность классификации, ниже которой объект не засчитывается",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    run_disabled = not field_files or ref_db.is_empty()
    if ref_db.is_empty():
        st.warning("Сначала загрузите хотя бы по несколько эталонных фото на каждый вид сорняка (шаг 1).")

    run_clicked = st.button("🔍 Запустить анализ", type="primary", disabled=run_disabled)

# ------------------------------------------------------------------------ #
# Обработка
# ------------------------------------------------------------------------ #
if run_clicked:
    with st.spinner("Строим индекс эталонов…"):
        ref_db.build_index()

    if not ref_db.ready:
        st.error("Не удалось построить индекс эталонов — проверьте загруженные фото.")
    else:
        known_species_now = sorted(ref_db.list_summary().keys())
        results: list[ImageResult] = []
        annotated_images: dict[str, np.ndarray] = {}
        all_crops: dict[str, np.ndarray] = {}
        species_order: list[str] = list(known_species_now)

        progress = st.progress(0.0, text="Обрабатываем фото…")
        for i, f in enumerate(field_files, start=1):
            bgr = read_image(f)
            if bgr is None:
                st.error(f"Не удалось прочитать файл {f.name}, пропускаю.")
                continue
            result, annotated, crops = process_image(
                bgr, f.name, ref_db, min_confidence=min_conf, species_order=species_order
            )
            results.append(result)
            annotated_images[f.name] = annotated
            all_crops.update(crops)
            progress.progress(i / len(field_files), text=f"Обработано {i}/{len(field_files)}: {f.name}")
        progress.empty()

        st.session_state["last_results"] = results
        st.session_state["last_annotated"] = annotated_images
        st.session_state["last_crops"] = all_crops
        st.session_state["last_species_order"] = species_order

# ------------------------------------------------------------------------ #
# Шаг 3. Результаты
# ------------------------------------------------------------------------ #
if "last_results" in st.session_state and st.session_state["last_results"]:
    results = st.session_state["last_results"]
    annotated_images = st.session_state["last_annotated"]
    all_crops = st.session_state.get("last_crops", {})
    species_order = st.session_state.get("last_species_order", [])

    with st.container(border=True):
        section_title("3", "Результаты")

        total_detections = sum(len(r.detections) for r in results)
        n_species = len({d.species for r in results for d in r.detections})
        n_images = len(results)

        m1, m2, m3 = st.columns(3)
        m1.metric("Всего найдено сорняков", total_detections)
        m2.metric("Обработано фото", n_images)
        m3.metric("Видов распознано", n_species)

        agg_df = aggregate_species_stage_counts(results, species_order=species_order)

        st.markdown("**Скачать результаты**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ results.json",
                data=results_to_json(results).encode("utf-8"),
                file_name="results.json",
                mime="application/json",
            )
        with c2:
            st.download_button(
                "⬇️ detections.csv",
                data=results_to_dataframe(results).to_csv(index=False).encode("utf-8-sig"),
                file_name="detections.csv",
                mime="text/csv",
            )
        with c3:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("results.json", results_to_json(results))
                zf.writestr("detections.csv", results_to_dataframe(results).to_csv(index=False))
                zf.writestr("summary.csv", agg_df.to_csv(index=False) if not agg_df.empty else "")
                for name, img in annotated_images.items():
                    zf.writestr(f"annotated/{name}", bgr_to_png_bytes(img))
                for rel_name, crop_img in all_crops.items():
                    zf.writestr(rel_name, bgra_to_png_bytes(crop_img))
            st.download_button(
                "⬇️ Всё архивом (.zip)",
                data=buf.getvalue(),
                file_name="weed_detection_results.zip",
                mime="application/zip",
            )

        if not agg_df.empty:
            colors = species_color_map(species_order or list(agg_df["вид"].unique()))
            domain = list(colors.keys())
            range_ = [colors[s] for s in domain]

            st.markdown("**Сводка по виду и стадии**")
            chart = (
                alt.Chart(agg_df)
                .mark_bar(cornerRadiusEnd=4, height=18)
                .encode(
                    y=alt.Y("метка:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
                    x=alt.X("количество:Q", title="Количество, шт."),
                    color=alt.Color(
                        "вид:N",
                        scale=alt.Scale(domain=domain, range=range_),
                        legend=alt.Legend(title="Вид сорняка"),
                    ),
                    tooltip=[
                        alt.Tooltip("вид:N", title="Вид"),
                        alt.Tooltip("стадия:N", title="Стадия"),
                        alt.Tooltip("количество:Q", title="Количество"),
                    ],
                )
                .properties(height=max(120, 46 * len(agg_df)))
                .configure_axis(grid=False, domainColor="#c3c2b7", labelColor="#52514e", titleColor="#52514e")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Таблица сводки (для проверки)"):
                st.dataframe(
                    agg_df.rename(columns={"вид": "Вид", "стадия": "Стадия", "количество": "Количество"})[
                        ["Вид", "Стадия", "Количество"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Сорняки с уверенностью выше порога не найдены — попробуйте снизить порог или добавить эталоны.")

        st.markdown("**Фото с разметкой**")
        for r in results:
            with st.expander(f"📷 {r.image_name} — найдено {len(r.detections)}", expanded=True):
                img_bgr = annotated_images[r.image_name]
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                if r.detections:
                    det_df = results_to_dataframe([r]).drop(columns=["image"])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Скачать это фото с рамками",
                    data=bgr_to_png_bytes(img_bgr),
                    file_name=f"annotated_{Path(r.image_name).stem}.png",
                    mime="image/png",
                    key=f"dl_img_{r.image_name}",
                )

                crops_for_image = [d for d in r.detections if d.crop_file and d.crop_file in all_crops]
                if crops_for_image:
                    st.markdown("**Отсечённые объекты (каждый сорняк отдельно, фон прозрачный)**")
                    thumb_cols = st.columns(min(6, len(crops_for_image)) or 1)
                    for j, d in enumerate(crops_for_image):
                        crop_bgra = all_crops[d.crop_file]
                        col = thumb_cols[j % len(thumb_cols)]
                        with col:
                            st.image(
                                cv2.cvtColor(crop_bgra, cv2.COLOR_BGRA2RGBA),
                                caption=f"#{d.id} {d.species}" + (f" · {d.stage}" if d.stage else ""),
                                use_container_width=True,
                            )
                            st.download_button(
                                "⬇️ PNG",
                                data=bgra_to_png_bytes(crop_bgra),
                                file_name=Path(d.crop_file).name,
                                mime="image/png",
                                key=f"dl_crop_{r.image_name}_{d.id}",
                            )
