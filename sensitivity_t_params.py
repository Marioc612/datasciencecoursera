"""sensitivity_t_params.py

Grid-search over tmin/topt/tmax/tlow per VPRM class.
For each combination, re-fits λ,PAR₀,α,β and records MSE.

Usage:
  python sensitivity_t_params.py \
    --config config.yaml \
    --csv csv_split_hybrid_FIXED_Merged.csv \
    --nc evi_lswi.nc \
    --split-col split --split-value train \
    --classes 1 2 5 6 \
    --outdir sensitivity_outputs
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def _bootstrap_local_repo_import() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyVPRM" / "VPRM.py").exists():
            pstr = str(parent)
            if pstr not in sys.path:
                sys.path.insert(0, pstr)
            return


_bootstrap_local_repo_import()

from scipy.optimize import curve_fit as _scipy_curve_fit
import pyVPRM.vprm_models.vprm_base_model as _bm_mod
from pyVPRM import VPRM as vprm_module
from pyVPRM.flux_tower_libs.flux_tower_class import flux_tower_data
from pyVPRM.sat_managers.base_manager import satellite_data_manager
from pyVPRM.vprm_models.vprm_base_model import vprm_base_model


def _robust_curve_fit(func, xdata, ydata, **kwargs):
    p0 = np.array(kwargs.get("p0", []), dtype=float) if "p0" in kwargs else None
    last_err = None
    for fev in [5000, 10000, 20000, 50000]:
        for _ in range(10):
            k = dict(kwargs)
            k["maxfev"] = fev
            if p0 is not None and p0.size > 0:
                jitter = np.random.normal(0.0, 0.20, size=p0.shape)
                k["p0"] = p0 * (1.0 + jitter)
            try:
                return _scipy_curve_fit(func, xdata, ydata, **k)
            except RuntimeError as e:
                last_err = e
    raise last_err


_bm_mod.curve_fit = _robust_curve_fit

VPRM_CLASS = (
    getattr(vprm_module, "vprm", None)
    or getattr(vprm_module, "vprm_preprocessor", None)
)
if VPRM_CLASS is None:
    raise ImportError("Could not find VPRM class in pyVPRM.VPRM")


# ── Re-use build/prepare helpers from your existing scripts ──────────────

def load_class_mappings(config_path: str):
    with open(config_path, "r") as f:
        vprm_cfg = yaml.safe_load(f)
    valid_vprm_classes = set()
    class_number_to_vprm = {}
    class_label_to_vprm = {}
    for _, cfg in vprm_cfg.items():
        vc = int(cfg["vprm_class"])
        valid_vprm_classes.add(vc)
        name = str(cfg.get("name", "")).strip().lower()
        if name:
            class_label_to_vprm[name] = vc
        for num in cfg.get("class_numbers", []):
            class_number_to_vprm[int(num)] = vc
    for label, cfg in vprm_cfg.items():
        class_label_to_vprm[str(label).strip().lower()] = int(cfg["vprm_class"])
    return class_number_to_vprm, valid_vprm_classes, class_label_to_vprm


def build_tower_instances(
    df, tower_col, time_col, lon_col, lat_col, class_col,
    tair_col, par_col, ssrd_col, nee_col, reco_col, use_reco,
    par_threshold_resp, class_number_to_vprm, valid_vprm_classes,
    class_label_to_vprm,
):
    """Identical to your fit script's build_tower_instances."""
    if tower_col not in df.columns:
        for cand in ["Tower", "Station", "site", "site_name"]:
            if cand in df.columns:
                tower_col = cand
                break
    required = {tower_col, time_col, lon_col, lat_col, class_col, tair_col, nee_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = df.copy()
    work[time_col] = (
        pd.to_datetime(work[time_col], utc=True)
        .dt.tz_convert("UTC").dt.tz_localize(None)
    )
    work = work.rename(columns={time_col: "datetime_utc", tair_col: "t2m"})

    if par_col in work.columns:
        work[par_col] = pd.to_numeric(work[par_col], errors="coerce")
        work.loc[work[par_col] <= par_threshold_resp, par_col] = 0.0
    if ssrd_col not in work.columns:
        if par_col not in work.columns:
            raise ValueError(f"Need '{ssrd_col}' or '{par_col}'")
        work[ssrd_col] = work[par_col] * 0.505
    if ssrd_col != "ssrd":
        work = work.rename(columns={ssrd_col: "ssrd"})

    numeric_cols = ["t2m", "ssrd", nee_col]
    if use_reco:
        numeric_cols.append(reco_col)
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work.loc[work[col] <= -9990, col] = pd.NA
    work = work.dropna(subset=["datetime_utc", "t2m", "ssrd", nee_col])
    if use_reco:
        work = work.dropna(subset=[reco_col])

    towers = []
    for tower, grp in work.groupby(tower_col):
        grp = grp.sort_values("datetime_utc").copy()
        classes = grp[class_col].dropna().unique()
        if len(classes) != 1:
            raise ValueError(f"{tower} has multiple classes: {classes}")
        raw_value = classes[0]
        raw_label = str(raw_value).strip().lower()
        land_type = None
        if raw_label in class_label_to_vprm:
            land_type = class_label_to_vprm[raw_label]
        else:
            try:
                raw_class = int(float(raw_value))
            except Exception:
                raw_class = None
            if raw_class is not None and raw_class in valid_vprm_classes:
                land_type = raw_class
            elif raw_class is not None and raw_class in class_number_to_vprm:
                land_type = class_number_to_vprm[raw_class]
        if land_type is None:
            raise ValueError(f"{tower} class '{raw_value}' could not be mapped")

        ft = flux_tower_data(
            t_start=grp["datetime_utc"].min(),
            t_stop=grp["datetime_utc"].max(),
            ssrd_key="ssrd", t2m_key="t2m",
            site_name=str(tower),
        )
        ft.lon = float(grp[lon_col].iloc[0])
        ft.lat = float(grp[lat_col].iloc[0])
        ft.set_land_type(int(land_type))

        keep = ["datetime_utc", "t2m", "ssrd", nee_col]
        rename_map = {nee_col: "NEE"}
        if use_reco and reco_col in grp.columns:
            keep.append(reco_col)
            rename_map[reco_col] = "RECO"
        ft.flux_data = grp[keep].rename(columns=rename_map).copy()
        towers.append(ft)
    return towers


def prepare_tower_nc(ds, tower_instances):
    """Identical to your existing prepare_tower_nc."""
    site_dim = next(
        (d for d in ["site_names", "site", "station", "Station", "sites"]
         if d in ds.dims), None
    )
    if site_dim is None:
        raise ValueError("NC needs site dimension")
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("NC needs time")
    for var in ["evi", "lswi"]:
        if var not in ds.data_vars:
            raise ValueError(f"NC missing '{var}'")
    if site_dim != "site_names":
        ds = ds.rename({site_dim: "site_names"})

    tower_names = [t.get_site_name() for t in tower_instances]
    existing = (
        [str(i) for i in ds["site_names"].values]
        if "site_names" in ds.coords
        else [str(i) for i in range(ds.sizes["site_names"])]
    )
    if set(existing) != set(tower_names):
        if "lat" in ds.coords and "lon" in ds.coords:
            lats, lons = ds["lat"].values, ds["lon"].values
            if ds.sizes["site_names"] < len(tower_names):
                raise ValueError("NC has fewer sites than towers")
            unused = set(range(ds.sizes["site_names"]))
            selected = []
            for tw in tower_instances:
                t_lon, t_lat = tw.get_lonlat()
                best = min(unused,
                           key=lambda i: (float(lons[i])-t_lon)**2 + (float(lats[i])-t_lat)**2)
                selected.append(best)
                unused.remove(best)
            ds = ds.isel(site_names=selected)
            ds = ds.assign_coords(site_names=("site_names", tower_names))
        else:
            if ds.sizes["site_names"] < len(tower_names):
                raise ValueError("NC has fewer sites than towers")
            ds = ds.isel(site_names=list(range(len(tower_names))))
            ds = ds.assign_coords(site_names=("site_names", tower_names))
    return ds


# ── Core: fit with a modified config and return MSE ─────────────────────

def fit_with_config(
    config_dict: dict,
    towers: list[flux_tower_data],
    nc_path: str,
    smoother: str,
    target_class: int,
) -> dict:
    """
    Write a temp config, run the full pyVPRM pipeline for ONE class,
    return fit_params + MSE for that class.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config_dict, tmp)
        tmp_config = tmp.name

    try:
        # Filter towers to target class only
        cls_towers = [t for t in towers if int(t.get_land_type()) == target_class]
        if not cls_towers:
            return {"fit_params": {}, "mse": float("nan"), "n": 0, "error": "no towers"}

        vp = VPRM_CLASS(vprm_config_path=tmp_config, flux_tower_instances=cls_towers)
        sat = xr.open_dataset(nc_path)
        sat = prepare_tower_nc(sat, tower_instances=cls_towers)
        sat_handler = satellite_data_manager(sat_img=sat)
        vp.add_sat_img(sat_handler,
                       mask_bad_pixels=False, mask_clouds=False,
                       mask_snow=False, mask_water=False)
        vp.sort_and_merge_by_timestamp()
        if smoother == "kalman":
            vp.kalman(keys=["evi", "lswi"], smooth_all=False)
        elif smoother == "lowess":
            vp.lowess(keys=["evi", "lswi"], smooth_all=False)
        vp.calc_min_max_evi_lswi()

        model = vprm_base_model(vprm_pre=vp, met=None)
        fit_ready = model.data_for_fitting()
        if not fit_ready:
            return {"fit_params": {}, "mse": float("nan"), "n": 0,
                    "error": "no data after prep"}

        fit_params = model.fit_vprm_data(
            data_list=fit_ready,
            variable_dict={"nee": "NEE"},
            fit_nee=True, fit_resp=True,
        )

        # Calculate MSE using the fitted params
        model2 = vprm_base_model(vprm_pre=None, met=None, fit_params_dict={
            int(k): v for k, v in fit_params.items()
        })

        all_obs, all_pred = [], []
        needed = ["Ps", "Ws", "Ts", "evi", "par", "tcorr"]
        for site in fit_ready:
            data = site.get_data()
            if data.empty:
                continue
            cls_int = int(site.get_land_type())
            if cls_int not in {int(k) for k in fit_params}:
                continue
            valid = data.dropna(subset=[c for c in needed if c in data.columns])
            if valid.empty:
                continue
            try:
                pred = model2.make_vprm_predictions(
                    inputs=valid[needed],
                    land_cover_type=cls_int,
                    concatenate_fluxes=True,
                )
                nee_pred = np.asarray(pred["nee"]).reshape(-1)
                nee_col = next((c for c in ["nee", "NEE"] if c in valid.columns), None)
                if nee_col is None:
                    continue
                nee_obs = valid[nee_col].values
                mask = np.isfinite(nee_pred) & np.isfinite(nee_obs)
                all_obs.extend(nee_obs[mask])
                all_pred.extend(nee_pred[mask])
            except Exception:
                continue

        if len(all_obs) == 0:
            return {"fit_params": fit_params, "mse": float("nan"), "n": 0,
                    "error": "no valid predictions"}

        all_obs = np.array(all_obs)
        all_pred = np.array(all_pred)
        mse = float(np.mean((all_obs - all_pred) ** 2))
        rmse = float(np.sqrt(mse))
        bias = float(np.mean(all_pred - all_obs))
        r2_ss_res = np.sum((all_obs - all_pred) ** 2)
        r2_ss_tot = np.sum((all_obs - np.mean(all_obs)) ** 2)
        r2 = float(1 - r2_ss_res / r2_ss_tot) if r2_ss_tot > 0 else float("nan")

        return {
            "fit_params": fit_params,
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "n": len(all_obs),
            "error": None,
        }

    except Exception as e:
        return {"fit_params": {}, "mse": float("nan"), "n": 0,
                "error": str(e)}
    finally:
        Path(tmp_config).unlink(missing_ok=True)


def generate_t_grid(current: dict, resolution: str = "coarse") -> list[dict]:
    """
    Generate grid of tmin/topt/tmax/tlow combinations.
    Respects physical constraints: tmin < tlow < topt < tmax
    """
    if resolution == "coarse":
        tmin_range = list(range(
            max(-5, current["tmin"] - 8),
            current["tmin"] + 10, 3
        ))
        topt_range = list(range(
            max(current["tmin"] + 5, current["topt"] - 10),
            current["topt"] + 12, 3
        ))
        tmax_range = list(range(
            max(current["topt"] + 5, current["tmax"] - 8),
            current["tmax"] + 10, 3
        ))
        tlow_range = list(range(
            max(-2, current["tlow"] - 8),
            current["tlow"] + 10, 3
        ))
    elif resolution == "fine":
        tmin_range = list(range(
            max(-5, current["tmin"] - 4),
            current["tmin"] + 5, 1
        ))
        topt_range = list(range(
            max(current["tmin"] + 3, current["topt"] - 5),
            current["topt"] + 6, 1
        ))
        tmax_range = list(range(
            max(current["topt"] + 3, current["tmax"] - 4),
            current["tmax"] + 5, 1
        ))
        tlow_range = list(range(
            max(-2, current["tlow"] - 4),
            current["tlow"] + 5, 1
        ))
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    combos = []
    for tmin, topt, tmax, tlow in itertools.product(
        tmin_range, topt_range, tmax_range, tlow_range
    ):
        # Physical constraints
        if tmin >= topt:
            continue
        if topt >= tmax:
            continue
        if tlow <= tmin:
            continue
        if tlow >= topt:
            continue
        combos.append({
            "tmin": tmin, "topt": topt, "tmax": tmax, "tlow": tlow
        })
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search tmin/topt/tmax/tlow per VPRM class"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--nc", required=True)
    parser.add_argument("--outdir", default="sensitivity_outputs")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Classes to test (default: all)")
    parser.add_argument("--resolution", choices=["coarse", "fine"], default="coarse",
                        help="Grid resolution: coarse (step=3) or fine (step=1)")
    parser.add_argument("--smoother", choices=["none", "kalman", "lowess"],
                        default="none")
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--split-value", default="train")
    parser.add_argument("--tower-col", default="Tower")
    parser.add_argument("--time-col", default="datetime_utc")
    parser.add_argument("--lon-col", default="Lon")
    parser.add_argument("--lat-col", default="Lat")
    parser.add_argument("--class-col", default="vprm_class")
    parser.add_argument("--tair-col", default="Tair")
    parser.add_argument("--par-col", default="PAR")
    parser.add_argument("--ssrd-col", default="ssrd")
    parser.add_argument("--nee-col", default="NEE")
    parser.add_argument("--reco-col", default="RECO")
    parser.add_argument("--par-threshold-resp", type=float, default=0.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    class_number_to_vprm, valid_vprm_classes, class_label_to_vprm = \
        load_class_mappings(args.config)

    # Load and filter CSV
    df = pd.read_csv(args.csv)
    if args.split_col:
        if args.split_col not in df.columns:
            raise ValueError(f"'{args.split_col}' not in CSV")
        df = df[df[args.split_col] == args.split_value].copy()
        if df.empty:
            raise ValueError(f"No rows for {args.split_col}='{args.split_value}'")

    # Build tower instances
    towers = build_tower_instances(
        df=df,
        tower_col=args.tower_col, time_col=args.time_col,
        lon_col=args.lon_col, lat_col=args.lat_col,
        class_col=args.class_col, tair_col=args.tair_col,
        par_col=args.par_col, ssrd_col=args.ssrd_col,
        nee_col=args.nee_col, reco_col=args.reco_col,
        use_reco=False,
        par_threshold_resp=args.par_threshold_resp,
        class_number_to_vprm=class_number_to_vprm,
        valid_vprm_classes=valid_vprm_classes,
        class_label_to_vprm=class_label_to_vprm,
    )

    # Determine which classes to search
    classes_available = sorted(set(int(t.get_land_type()) for t in towers))
    if args.classes:
        target_classes = [c for c in args.classes if c in classes_available]
    else:
        target_classes = classes_available

    print(f"Target classes: {target_classes}")
    print(f"Resolution: {args.resolution}")

    # Map vprm_class → config key
    class_to_config_key = {}
    for key, cfg in base_config.items():
        class_to_config_key[int(cfg["vprm_class"])] = key

    # ── Main grid search loop ──
    all_results = []

    for cls in target_classes:
        config_key = class_to_config_key.get(cls)
        if config_key is None:
            print(f"[warn] class {cls} not in config, skipping")
            continue

        current_t = {
            "tmin": base_config[config_key]["tmin"],
            "topt": base_config[config_key]["topt"],
            "tmax": base_config[config_key]["tmax"],
            "tlow": base_config[config_key]["tlow"],
        }

        print(f"\n{'='*60}")
        print(f"CLASS {cls} ({config_key})")
        print(f"Current: tmin={current_t['tmin']}, topt={current_t['topt']}, "
              f"tmax={current_t['tmax']}, tlow={current_t['tlow']}")

        combos = generate_t_grid(current_t, resolution=args.resolution)
        print(f"Testing {len(combos)} combinations...")

        # First: baseline (current config)
        print(f"  [baseline] ...", end="", flush=True)
        baseline = fit_with_config(
            config_dict=base_config,
            towers=towers,
            nc_path=args.nc,
            smoother=args.smoother,
            target_class=cls,
        )
        print(f" MSE={baseline['mse']:.4f}, R²={baseline.get('r2', float('nan')):.4f}, "
              f"n={baseline['n']}")

        best_mse = baseline["mse"]
        best_combo = current_t.copy()
        best_result = baseline

        for i, combo in enumerate(combos):
            # Modify config for this class only
            test_config = copy.deepcopy(base_config)
            test_config[config_key]["tmin"] = combo["tmin"]
            test_config[config_key]["topt"] = combo["topt"]
            test_config[config_key]["tmax"] = combo["tmax"]
            test_config[config_key]["tlow"] = combo["tlow"]

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(combos)}] tmin={combo['tmin']}, "
                      f"topt={combo['topt']}, tmax={combo['tmax']}, "
                      f"tlow={combo['tlow']} ...", end="", flush=True)

            try:
                result = fit_with_config(
                    config_dict=test_config,
                    towers=towers,
                    nc_path=args.nc,
                    smoother=args.smoother,
                    target_class=cls,
                )
            except Exception as e:
                result = {"mse": float("nan"), "n": 0, "error": str(e)}

            row = {
                "vprm_class": cls,
                "config_key": config_key,
                **combo,
                "mse": result.get("mse", float("nan")),
                "rmse": result.get("rmse", float("nan")),
                "bias": result.get("bias", float("nan")),
                "r2": result.get("r2", float("nan")),
                "n": result.get("n", 0),
                "error": result.get("error"),
            }
            # Store fitted params if available
            fp = result.get("fit_params", {})
            if fp and str(cls) in fp:
                for pk, pv in fp[str(cls)].items():
                    row[f"param_{pk}"] = pv
            elif fp:
                for k, v in fp.items():
                    if isinstance(v, dict):
                        for pk, pv in v.items():
                            row[f"param_{pk}"] = pv
                        break

            all_results.append(row)

            if (i + 1) % 20 == 0 or i == 0:
                print(f" MSE={row['mse']:.4f}, R²={row['r2']:.4f}")

            if np.isfinite(result.get("mse", float("nan"))):
                if result["mse"] < best_mse or not np.isfinite(best_mse):
                    best_mse = result["mse"]
                    best_combo = combo.copy()
                    best_result = result

        # Report best
        print(f"\n  ── BEST for class {cls} ──")
        print(f"  tmin={best_combo['tmin']}, topt={best_combo['topt']}, "
              f"tmax={best_combo['tmax']}, tlow={best_combo['tlow']}")
        print(f"  MSE={best_mse:.4f} (baseline={baseline['mse']:.4f}, "
              f"improvement={100*(baseline['mse']-best_mse)/baseline['mse']:.1f}%)")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "sensitivity_t_params.csv", index=False)

    # Save best per class
    best_per_class = []
    for cls in target_classes:
        cls_df = results_df[results_df["vprm_class"] == cls].copy()
        if cls_df.empty:
            continue
        best_row = cls_df.loc[cls_df["mse"].idxmin()] if cls_df["mse"].notna().any() else None
        if best_row is not None:
            best_per_class.append(best_row.to_dict())

    pd.DataFrame(best_per_class).to_csv(
        outdir / "best_t_params_per_class.csv", index=False
    )

    # Generate improved config
    improved_config = copy.deepcopy(base_config)
    for row in best_per_class:
        config_key = row["config_key"]
        improved_config[config_key]["tmin"] = int(row["tmin"])
        improved_config[config_key]["topt"] = int(row["topt"])
        improved_config[config_key]["tmax"] = int(row["tmax"])
        improved_config[config_key]["tlow"] = int(row["tlow"])

    with open(outdir / "config_improved.yaml", "w") as f:
        yaml.dump(improved_config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Saved: {outdir / 'sensitivity_t_params.csv'}")
    print(f"Saved: {outdir / 'best_t_params_per_class.csv'}")
    print(f"Saved: {outdir / 'config_improved.yaml'}")


if __name__ == "__main__":
    main()