#!/usr/bin/env python3
"""저장된 scoring function 으로 pose 묶음에서 Top-k 를 고릅니다. (독립 실행)

필요한 것: xgboost, numpy, pandas. 그리고 이 폴더의 model.json / feature_order.json /
spec.json. 28~36번 파이프라인 스크립트는 **필요하지 않습니다.**

2단 구조입니다.
  1단  feature -> 점수      model.json (rank:ndcg 로 학습된 트리)
  2단  점수 -> Top-k        규칙 A (생성기별 1위 먼저, 남은 자리는 점수순)

쓰는 법
  python predict.py --poses poses.parquet --k 5 --out top5.csv
  python predict.py --poses poses.csv --k 3
  python predict.py --poses poses.parquet --ensemble      # 시드 5개 평균(참고용)

poses 파일에 있어야 하는 열
  system_id   타깃 식별자. 이 열로 묶어 타깃마다 Top-k 를 고릅니다
  model       생성기 이름(af3/bt2/of3/pt2 등). 규칙 A 가 이 열로 다양성을 셉니다
  pose_id     동점을 깨는 키. SHA1 오름차순으로 깹니다
  seed sample 있으면 결과에 함께 싣습니다(없어도 됩니다)
  + feature_order.json 의 feature 열 전부

주의
  결측은 빈 값으로 두십시오. 0 으로 채우면 틀린 점수가 납니다
  (학습 때 NaN 을 NaN 으로 다뤘기 때문입니다).
  열 순서는 이 스크립트가 feature_order.json 을 보고 맞춥니다. 직접 맞추지 마십시오.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost


def tie_key(pose_id):
    """동점을 결정적으로 깨되 생성기와 무관한 키. 학습 단계와 같은 함수입니다."""
    return int(hashlib.sha1(str(pose_id).encode()).hexdigest(), 16)


def rule_a_order(generators, kmax):
    """규칙 A — 점수 내림차순 배열에서 고를 위치를 순서대로 돌려줍니다.

    생성기가 처음 나오면 먼저 담고, 이미 쓴 생성기는 뒤로 보냅니다. 그 순서의 앞 kmax개.
    31번 rule_order 의 ruleA 와 같은 동작입니다.
    """
    seen, first, rest = set(), [], []
    for i, gen in enumerate(generators):
        if gen in seen:
            rest.append(i)
        else:
            seen.add(gen)
            first.append(i)
    return (first + rest)[:kmax]


def load_bundle(model_dir, ensemble):
    """모델과 feature 순서, 규칙 이름을 읽습니다."""
    model_dir = Path(model_dir)
    order_path = model_dir / "feature_order.json"
    spec_path = model_dir / "spec.json"
    for path in (order_path, spec_path):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    features = json.loads(order_path.read_text(encoding="utf-8"))["features"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rule = spec["scoring_function"]["stage2_rule"]["name"]
    if rule != "ruleA":
        raise ValueError(f"이 스크립트는 규칙 A 만 구현합니다(spec: {rule}).")
    gen_col = spec["scoring_function"]["stage2_rule"].get("generator_column", "model")
    if ensemble:
        paths = sorted(model_dir.glob("model_seed*.json"))
        if not paths:
            raise FileNotFoundError("model_seed*.json 이 없습니다.")
    else:
        paths = [model_dir / "model.json"]
        if not paths[0].exists():
            raise FileNotFoundError(f"model.json 이 없습니다: {paths[0]}")
    boosters = []
    for path in paths:
        booster = xgboost.Booster()
        booster.load_model(str(path))
        boosters.append(booster)
    return features, boosters, [p.name for p in paths], gen_col


def read_poses(path):
    """parquet 또는 csv 를 읽습니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pose 파일이 없습니다: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def score_poses(frame, features, boosters):
    """1단 — feature 행렬을 사전에 정한 순서로 만들어 점수를 냅니다."""
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"pose 파일에 없는 feature 열: {missing}")
    matrix = frame[features].to_numpy(dtype=np.float32, na_value=np.nan)
    dmat = xgboost.DMatrix(matrix)
    parts = [booster.predict(dmat) for booster in boosters]
    return np.mean(np.vstack(parts), axis=0) if len(parts) > 1 else parts[0]


def pick_topk(frame, scores, k, gen_col):
    """2단 — 타깃마다 규칙 A 로 상위 k개를 고릅니다."""
    needed = ["system_id", gen_col, "pose_id"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"pose 파일에 없는 열: {missing}")
    extra = [c for c in ("seed", "sample") if c in frame.columns]
    ties = frame["pose_id"].map(tie_key).to_numpy()
    rows = []
    for system, idx in frame.groupby("system_id", sort=False).indices.items():
        idx = np.asarray(idx)
        chunk = np.where(np.isnan(scores[idx]), -np.inf, scores[idx])
        order = np.lexsort((ties[idx], -chunk))
        gens = frame[gen_col].to_numpy()[idx][order]
        for rank, pos in enumerate(rule_a_order(list(gens), k), start=1):
            row_at = idx[order[pos]]
            item = {"system_id": str(system), "rank": rank,
                    "pose_id": str(frame["pose_id"].to_numpy()[row_at]),
                    "generator": str(frame[gen_col].to_numpy()[row_at]),
                    "score": round(float(scores[row_at]), 6)}
            for col in extra:
                item[col] = frame[col].to_numpy()[row_at]
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="pose 표 (parquet 또는 csv).")
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent),
                        help="model.json 등이 있는 폴더. 기본은 이 스크립트 폴더.")
    parser.add_argument("--k", type=int, default=5, help="타깃마다 고를 pose 수.")
    parser.add_argument("--out", default=None, help="결과 csv 경로. 없으면 화면에 씁니다.")
    parser.add_argument("--ensemble", action="store_true",
                        help="시드 5개 평균을 씁니다(참고용. 출하 모델은 단일입니다).")
    parser.add_argument("--scores-out", default=None,
                        help="pose 별 점수 전체를 쓸 csv 경로(선택).")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k 는 1 이상이어야 합니다.")

    features, boosters, used, gen_col = load_bundle(args.model_dir, args.ensemble)
    frame = read_poses(args.poses)
    print(f"모델: {used}", file=sys.stderr)
    print(f"feature {len(features)}개 | pose {len(frame)}행 "
          f"| 타깃 {frame['system_id'].nunique()}개", file=sys.stderr)

    scores = score_poses(frame, features, boosters)
    nan_count = int(np.isnan(scores).sum())
    if nan_count:
        print(f"경고: 점수가 NaN 인 pose {nan_count}개 (가장 뒤로 보냅니다)",
              file=sys.stderr)
    rows = pick_topk(frame, scores, args.k, gen_col)

    if args.scores_out:
        dump = frame[[c for c in ("system_id", gen_col, "seed", "sample", "pose_id")
                      if c in frame.columns]].copy()
        dump["score"] = scores
        dump.to_csv(args.scores_out, index=False, encoding="utf-8")
        print(f"점수 전체: {args.scores_out}", file=sys.stderr)

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False, encoding="utf-8")
        print(f"Top-{args.k}: {args.out} ({len(table)}행)", file=sys.stderr)
    else:
        table.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
