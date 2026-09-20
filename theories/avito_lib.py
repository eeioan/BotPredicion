"""Общий код для всех ноутбуков проекта: фильтр окна и построение признаков.

Вынесено в модуль, чтобы три ноутбука не держали три копии одной логики. Если копии
разойдутся, ресерч начнёт изучать не те данные, на которых учится модель, — а заметить
это по цифрам невозможно.

Все функции работают только с событиями внутри окна [window_start_ts, window_end_ts).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]

# самые «сырые» признаки — примерно то, с чего начинает quickstart
RAW_FEATURES = ["n_events", "n_items_uniq", "n_cat_uniq", "n_loc_uniq",
                "n_query_uniq", "events_per_hour"]

SCRAPER_CLIENTS = ["Scrapy", "curl", "node-fetch", "urllib3", "requests", "Go-http-client"]
UA_FAMILIES = ["chrome", "firefox", "safari", "yabrowser", "avito_app", "client"]
UA_OSES = ["windows", "macos", "linux_x11", "android", "ios"]
PLATFORMS = ["desktop", "web", "android", "ios"]
TRANSITIONS = [
    "search_results_view>search_results_view",
    "search_results_view>item_view",
    "item_view>photo_swipe",
    "item_view>item_view",
    "item_view>contact_phone_show",
    "item_view>seller_page_view",
]


def load_data(data_dir="data"):
    """Читает train/test/events с разбором дат."""
    dates = ["cookie_created_at", "window_start_ts", "window_end_ts"]
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=dates)
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=dates)
    events = pd.read_csv(f"{data_dir}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def normalize_platform(s):
    """WEB/Web/web -> web, iphone/IOS/iOS/ios -> ios. Без этого счётчики по платформе — мусор."""
    return s.astype("string").str.lower().replace({"iphone": "ios"})


def clip_to_window(events_raw, meta):
    """Единственная дверь к событиям: оставляет только то, что внутри окна строки.

    Всё, что за правой границей окна, — информация из будущего. Assert внутри гарантирует,
    что captcha_shown (она вся лежит после окна) физически не попадёт в признаки.
    """
    ev = events_raw.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner"
    )
    keep = (ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])
    ev = ev.loc[keep].copy()
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)
    ev["platform_norm"] = normalize_platform(ev["platform"])
    assert (ev["event_name"] == "captcha_shown").sum() == 0, "после фильтра осталась captcha_shown"
    return ev


# --------------------------------------------------------------------------- user_agent


def _ua_family(s):
    for c in SCRAPER_CLIENTS:
        if c in s:
            return "client"
    if "okhttp" in s:
        return "avito_app"
    if "YaBrowser" in s:
        return "yabrowser"
    if "Firefox" in s:
        return "firefox"
    if "Chrome" in s:
        return "chrome"
    if "Safari" in s:
        return "safari"
    return "other"


def _ua_os(s):
    if "Windows NT" in s:
        return "windows"
    if "Macintosh" in s or "Mac OS X" in s:
        return "macos"
    if "X11" in s:
        return "linux_x11"
    if "Android" in s:
        return "android"
    if "iPhone" in s or "iPad" in s:
        return "ios"
    return "other"


def _ua_version(s):
    for pat in [r"Chrome/(\d+)", r"Firefox/(\d+)", r"Version/(\d+)", r"Avito/(\d+)",
                r"Scrapy/(\d+)", r"curl/(\d+)", r"node-fetch/(\d+)",
                r"urllib3/(\d+)", r"requests/(\d+)", r"Go-http-client/(\d+)"]:
        m = re.search(pat, s)
        if m:
            return float(m.group(1))
    return np.nan


def build_population_stats(events, train, test):
    """Популяционные статистики СТРОГО по прошлым дням.

    Популярность товара и частота user_agent считаются накопительно: для строки за
    конкретный день берутся только события предыдущих дней. Если считать по всем данным
    сразу, в признак попадёт будущее — проверено, даёт +0.14 несуществующего качества.
    """
    ev_all = pd.concat([clip_to_window(events, train), clip_to_window(events, test)],
                       ignore_index=True)

    pairs = ev_all.dropna(subset=["item_id"])[["item_id", "cookie_id", "window_start_ts"]]
    pairs = pairs.drop_duplicates()
    per_day = pairs.groupby(["item_id", "window_start_ts"]).size().unstack(fill_value=0)
    per_day = per_day.sort_index(axis=1)
    item_pop_before = per_day.cumsum(axis=1).shift(1, axis=1).fillna(0).stack()

    ua_pairs = ev_all[["user_agent", "cookie_id", "window_start_ts"]].drop_duplicates()
    ua_day = ua_pairs.groupby(["user_agent", "window_start_ts"]).size().unstack(fill_value=0)
    ua_day = ua_day.sort_index(axis=1)
    ua_before = ua_day.cumsum(axis=1).shift(1, axis=1).fillna(0)
    ua_freq_before = (ua_before / ua_before.sum(axis=0).replace(0, np.nan)).stack()

    ua = pd.Series(events["user_agent"].unique())
    ua_info = pd.DataFrame({
        "user_agent": ua.values,
        "ua_family": [_ua_family(s) for s in ua],
        "ua_os": [_ua_os(s) for s in ua],
        "ua_ver": [_ua_version(s) for s in ua],
    }).set_index("user_agent")
    ua_info["ua_is_client"] = (ua_info["ua_family"] == "client").astype(float)

    return {"item_pop": item_pop_before, "ua_freq": ua_freq_before, "ua_info": ua_info}


# --------------------------------------------------------------------------- признаки


def _ent(s):
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p)).sum()) if len(p) else np.nan


def build_features(meta, events_raw, pop, progress=None):
    """Считает все признаки. Возвращает (DataFrame, словарь группа -> колонки).

    Одна функция на train и test — это гарантия, что наборы не разъедутся.
    Порядок строк совпадает с meta.
    """
    ev = clip_to_window(events_raw, meta)
    idx = pd.Index(meta["cookie_id"].values, name="cookie_id")
    F = pd.DataFrame(index=idx)
    groups = {}
    g = ev.groupby("cookie_id", sort=False)
    win_hours = (meta["window_end_ts"] - meta["window_start_ts"]).dt.total_seconds().values / 3600.0
    cookie_day = pd.Series(meta["window_start_ts"].values, index=meta["cookie_id"].values)

    def put(grp, name, val):
        F[name] = val.reindex(idx) if isinstance(val, pd.Series) else val
        groups.setdefault(grp, []).append(name)

    def tick(msg):
        if progress is not None:
            progress(msg)

    # ---------- база ----------
    tick("объём")
    put("base", "n_events", g.size())
    F["n_events"] = F["n_events"].fillna(0)
    put("base", "n_items_uniq", g["item_id"].nunique())
    put("base", "n_cat_uniq", g["item_category"].nunique())
    put("base", "n_loc_uniq", g["item_location"].nunique())
    put("base", "n_query_uniq", g["search_query"].nunique())
    put("base", "events_per_hour", F["n_events"].values / win_hours)

    cnt = pd.crosstab(ev["cookie_id"], ev["event_name"]).reindex(columns=EVENT_NAMES, fill_value=0)
    cnt = cnt.reindex(idx).fillna(0.0)
    for c in EVENT_NAMES:
        put("base", "cnt_" + c, cnt[c])

    tick("ритм базовый")
    ev = ev.copy()
    ev["dt"] = g["event_ts"].diff().dt.total_seconds()
    dtd = ev.dropna(subset=["dt"])
    gd = dtd.groupby("cookie_id")["dt"]
    put("base", "dt_median", gd.median())
    put("base", "dt_min", gd.min())
    put("base", "dt_std", gd.std())
    put("base", "dt_frac_lt_1s", gd.apply(lambda s: (s < 1).mean()))
    put("base", "span_seconds", (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds())
    put("base", "n_active_hours", ev.assign(h=ev["event_ts"].dt.hour).groupby("cookie_id")["h"].nunique())

    views = F["cnt_item_view"].clip(lower=1)
    searches = F["cnt_search_results_view"].clip(lower=1)
    contacts = F["cnt_contact_phone_show"] + F["cnt_contact_chat_open"] + F["cnt_contact_message_sent"]
    put("base", "photo_per_view", F["cnt_photo_swipe"] / views)
    put("base", "contact_per_view", contacts / views)
    put("base", "view_per_search", F["cnt_item_view"] / searches)
    put("base", "max_search_page", g["search_page"].max())
    put("base", "mean_search_page", g["search_page"].mean())
    put("base", "ptr_frac", g["pointer_x"].apply(lambda s: s.notna().mean()))
    put("base", "ptr_x_nunique", g["pointer_x"].nunique())
    xy = ev.dropna(subset=["pointer_x", "pointer_y"]).copy()
    xy["_xy"] = xy["pointer_x"].astype(int).astype(str) + "_" + xy["pointer_y"].astype(int).astype(str)
    put("base", "ptr_xy_nunique", xy.groupby("cookie_id")["_xy"].nunique())
    F["ptr_xy_nunique"] = F["ptr_xy_nunique"].fillna(0.0)
    put("base", "n_platform_norm", g["platform_norm"].nunique())
    put("base", "n_user_agent", g["user_agent"].nunique())
    put("base", "cookie_age_days",
        (meta["window_end_ts"] - meta["cookie_created_at"]).dt.total_seconds().values / 86400.0)
    put("base", "window_dow", meta["window_start_ts"].dt.dayofweek.values.astype(float))

    # ---------- content ----------
    tick("содержимое")
    evc = ev.dropna(subset=["item_category"])
    put("content", "cat_entropy", evc.groupby("cookie_id")["item_category"].apply(_ent))
    put("content", "cat_top_frac",
        evc.groupby("cookie_id")["item_category"].apply(lambda s: s.value_counts(normalize=True).max()))
    evc = evc.copy()
    evc["prev"] = evc.groupby("cookie_id")["item_category"].shift()
    put("content", "cat_switch_frac",
        evc.assign(sw=((evc["prev"].notna()) & (evc["prev"] != evc["item_category"])).astype(float))
           .groupby("cookie_id")["sw"].mean())
    evl = ev.dropna(subset=["item_location"])
    put("content", "loc_entropy", evl.groupby("cookie_id")["item_location"].apply(_ent))
    put("content", "loc_top_frac",
        evl.groupby("cookie_id")["item_location"].apply(lambda s: s.value_counts(normalize=True).max()))
    evi = ev.dropna(subset=["item_id"])
    gi = evi.groupby("cookie_id")
    put("content", "item_repeat_frac", 1 - gi["item_id"].nunique() / gi.size())
    _ik = pd.MultiIndex.from_arrays([evi["item_id"].values, evi["window_start_ts"].values])
    put("content", "item_pop_mean",
        evi.assign(p=pop["item_pop"].reindex(_ik).values).groupby("cookie_id")["p"].mean())
    stt = pd.crosstab(ev["cookie_id"], ev["seller_type"], normalize="index")
    put("content", "seller_frac_private",
        stt["private"] if "private" in stt.columns else pd.Series(0.0, index=stt.index))
    put("content", "seller_known_frac", g["seller_type"].apply(lambda s: s.notna().mean()))

    # ---------- rhythm ----------
    tick("ритм расширенный")
    put("rhythm", "dt_mean", gd.mean())
    put("rhythm", "dt_max", gd.max())
    put("rhythm", "dt_p10", gd.quantile(0.10))
    put("rhythm", "dt_p90", gd.quantile(0.90))
    put("rhythm", "dt_frac_lt_5s", gd.apply(lambda s: (s < 5).mean()))
    F["dt_cv"] = F["dt_std"] / F["dt_mean"].clip(lower=1e-9)
    groups["rhythm"].append("dt_cv")
    ev["new_sess"] = ev["dt"].isna() | (ev["dt"] > 1800)
    ev["sess_id"] = ev.groupby("cookie_id")["new_sess"].cumsum()
    sess = ev.groupby(["cookie_id", "sess_id"]).size().rename("n").reset_index()
    put("rhythm", "n_sessions", sess.groupby("cookie_id")["sess_id"].max())
    put("rhythm", "sess_events_mean", sess.groupby("cookie_id")["n"].mean())
    put("rhythm", "sess_events_max", sess.groupby("cookie_id")["n"].max())
    put("rhythm", "hour_entropy", ev.assign(h=ev["event_ts"].dt.hour).groupby("cookie_id")["h"].apply(_ent))
    put("rhythm", "night_frac",
        ev.assign(n=(ev["event_ts"].dt.hour < 6).astype(float)).groupby("cookie_id")["n"].mean())

    # ---------- navquery ----------
    tick("навигация и запросы")
    srch = ev[ev["event_name"] == "search_results_view"].copy()
    gs = srch.groupby("cookie_id")
    put("navquery", "page_gt5_frac", gs["search_page"].apply(lambda s: (s > 5).mean()))
    put("navquery", "page_gt10_frac", gs["search_page"].apply(lambda s: (s > 10).mean()))
    put("navquery", "page_std", gs["search_page"].std())
    srch["prev_page"] = gs["search_page"].shift()
    put("navquery", "page_step_plus1_frac",
        srch.assign(st=((srch["search_page"] - srch["prev_page"]) == 1).astype(float))
            .groupby("cookie_id")["st"].mean())
    n_srch = gs.size()
    n_q = gs["search_query"].nunique()
    put("navquery", "q_repeat_frac", 1 - n_q / n_srch.clip(lower=1))
    put("navquery", "q_len_mean", gs["search_query"].apply(lambda s: s.dropna().str.len().mean()))

    # ---------- pointer ----------
    tick("указатель")
    pxy = ev.dropna(subset=["pointer_x", "pointer_y"]).copy()
    pxy["dist"] = np.sqrt(pxy.groupby("cookie_id")["pointer_x"].diff() ** 2
                          + pxy.groupby("cookie_id")["pointer_y"].diff() ** 2)
    pxy["_xy"] = pxy["pointer_x"].astype(int).astype(str) + "_" + pxy["pointer_y"].astype(int).astype(str)
    gp = pxy.groupby("cookie_id")
    put("pointer", "ptr_x_std", gp["pointer_x"].std())
    put("pointer", "ptr_y_std", gp["pointer_y"].std())
    put("pointer", "ptr_dist_mean", gp["dist"].mean())
    put("pointer", "ptr_dist_std", gp["dist"].std())
    put("pointer", "ptr_repeat_frac", 1 - gp["_xy"].nunique() / gp.size())

    # ---------- sequence ----------
    tick("порядок событий")
    sq = ev[["cookie_id", "event_name"]].copy()
    sq["nxt"] = sq.groupby("cookie_id")["event_name"].shift(-1)
    sq = sq.dropna(subset=["nxt"])
    sq["pair"] = sq["event_name"] + ">" + sq["nxt"]
    tot = sq.groupby("cookie_id").size()
    pv = pd.crosstab(sq["cookie_id"], sq["pair"])
    for pair in TRANSITIONS:
        col = pv[pair] if pair in pv.columns else pd.Series(0.0, index=pv.index)
        put("sequence", "tr_" + pair.replace(">", "_to_"), col / tot)
    put("sequence", "funnel_depth",
        (F["cnt_search_results_view"] > 0).astype(int) + (F["cnt_item_view"] > 0).astype(int)
        + (F["cnt_photo_swipe"] > 0).astype(int) + (contacts > 0).astype(int))

    # ---------- uaplat ----------
    tick("user_agent и платформа")
    ua_ck = ev.groupby("cookie_id")["user_agent"].agg(lambda s: s.mode().iat[0])
    uj = pop["ua_info"].reindex(ua_ck.values)
    put("uaplat", "ua_is_client", pd.Series(uj["ua_is_client"].values, index=ua_ck.index))
    put("uaplat", "ua_ver", pd.Series(uj["ua_ver"].values, index=ua_ck.index))
    _uk = pd.MultiIndex.from_arrays([ua_ck.values, cookie_day.reindex(ua_ck.index).values])
    put("uaplat", "ua_freq", pd.Series(pop["ua_freq"].reindex(_uk).values, index=ua_ck.index))
    for fam in UA_FAMILIES:
        put("uaplat", "ua_fam_" + fam,
            pd.Series((uj["ua_family"].values == fam).astype(float), index=ua_ck.index))
    for os_ in UA_OSES:
        put("uaplat", "ua_os_" + os_,
            pd.Series((uj["ua_os"].values == os_).astype(float), index=ua_ck.index))
    pl = pd.crosstab(ev["cookie_id"], ev["platform_norm"], normalize="index")
    for p in PLATFORMS:
        put("uaplat", "plat_frac_" + p, pl[p] if p in pl.columns else pd.Series(0.0, index=pl.index))
    ev["prev_plat"] = ev.groupby("cookie_id")["platform_norm"].shift()
    put("uaplat", "plat_switch_frac",
        ev.assign(sw=((ev["prev_plat"].notna()) & (ev["prev_plat"] != ev["platform_norm"])).astype(float))
          .groupby("cookie_id")["sw"].mean())

    zero = ["n_events", "events_per_hour", "ptr_xy_nunique"] + ["cnt_" + c for c in EVENT_NAMES]
    F[zero] = F[zero].fillna(0.0)
    return F.reset_index(drop=True), groups


# --------------------------------------------------------- связи между куками


def build_cov_index(events, train, test):
    """Кто с кем пересекался по просмотренным объявлениям, СТРОГО по прошлым дням.

    Для куки за день D учитываются только куки, чьё окно началось раньше D. Иначе
    признак заглядывал бы в будущее — проверено, даёт несуществующее качество.

    Возвращает словарь cookie_id -> Counter{другая кука: сколько общих товаров}.
    """
    from collections import Counter, defaultdict

    ev = pd.concat([clip_to_window(events, train), clip_to_window(events, test)])
    pairs = ev.dropna(subset=["item_id"])[["cookie_id", "item_id", "window_start_ts"]]
    pairs = pairs.drop_duplicates()

    part = defaultdict(Counter)
    for _, grp in pairs.groupby("item_id"):
        ck, dy = grp["cookie_id"].values, grp["window_start_ts"].values
        # слишком популярные объявления связывают всех со всеми и только шумят
        if len(ck) < 2 or len(ck) > 60:
            continue
        for i in range(len(ck)):
            for j in range(len(ck)):
                if i != j and dy[j] < dy[i]:
                    part[ck[i]][ck[j]] += 1
    return part


def covisitation_features(meta, cov_index):
    """Три признака из графа пересечений. Порядок строк — как в meta."""
    idx = pd.Index(meta["cookie_id"].values)
    n = pd.Series({k: len(v) for k, v in cov_index.items()})
    mx = pd.Series({k: max(v.values()) for k, v in cov_index.items()})
    st = pd.Series({k: sum(1 for x in v.values() if x >= 2) for k, v in cov_index.items()})
    return pd.DataFrame({
        "cov_partners": n.reindex(idx).fillna(0).values,
        "cov_max_shared": mx.reindex(idx).fillna(0).values,
        "cov_strong": st.reindex(idx).fillna(0).values,
    })


# --------------------------------------------------- кодировка объявлений метками

TE_COLS = ["item_te_mean", "item_te_max", "item_known_frac"]


def item_pairs(meta, events):
    """Пары (номер строки, объявление) для кодировки."""
    ev = clip_to_window(events, meta).dropna(subset=["item_id"])
    p = ev[["cookie_id", "item_id"]].drop_duplicates()
    pos = pd.Series(np.arange(len(meta)), index=meta["cookie_id"].values)
    p = p.assign(row=pos.reindex(p["cookie_id"]).values)
    return p[["row", "item_id"]]


def encode_items(fit_pairs, fit_y, apply_pairs, n_apply, k=10.0, drop_own=True):
    """Доля ботов среди кук, смотревших объявление, со сглаживанием к базовой доле.

    Статистика строится ТОЛЬКО по fit_pairs. Если применяем к тем же строкам, на
    которых учились (drop_own=True), собственный вклад строки вычитается — иначе
    признак предсказывал бы метку, из которой сам и сделан.

    Возвращает массив (n_apply, 3): среднее, максимум и доля известных объявлений.
    """
    # базовая доля считается ТОЛЬКО по обучающим строкам: иначе в сглаживание
    # просочится доля ботов из валидации
    fit_rows = np.unique(fit_pairs["row"].values)
    p = float(np.mean(np.asarray(fit_y)[fit_rows]))
    fp = fit_pairs.assign(y=np.asarray(fit_y)[fit_pairs["row"].values])
    st = fp.groupby("item_id")["y"].agg(["sum", "size"])

    ap = apply_pairs
    s = ap["item_id"].map(st["sum"]).fillna(0.0).values
    n = ap["item_id"].map(st["size"]).fillna(0.0).values

    if drop_own:
        own = np.asarray(fit_y)[ap["row"].values].astype(float)
        s, n = s - own, n - 1.0

    enc = (s + k * p) / (n + k)
    d = pd.DataFrame({"row": ap["row"].values, "enc": enc, "known": (n > 0).astype(float)})
    g = d.groupby("row")

    out = np.full((n_apply, 3), np.nan)
    out[g["enc"].mean().index.values, 0] = g["enc"].mean().values
    out[g["enc"].max().index.values, 1] = g["enc"].max().values
    out[g["known"].mean().index.values, 2] = g["known"].mean().values
    return out
