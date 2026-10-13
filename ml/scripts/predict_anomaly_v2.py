"""Apply the selected v2 detector to new five-second sensor records."""
from pathlib import Path
import argparse
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import pandas as pd
from manufacturing_ai.anomaly import FEATURES, KEYS
from manufacturing_ai.anomaly_v2 import load_selected, predict_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default=str(ROOT / 'models/anomaly_v2/selected_model.joblib'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/anomaly_v2/new_observations.csv'))
    parser.add_argument('--max-rows', type=int, help='Limit input only for an explicit smoke test')
    args = parser.parse_args()
    source, destination = Path(args.input).resolve(), Path(args.output).resolve()
    if ROOT not in destination.parents:
        raise ValueError('Output must remain under this ml directory')
    if destination == source or destination == Path(args.model).resolve():
        raise ValueError('Output must not overwrite input data or a model')
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError('--max-rows must be positive')
    bundle = load_selected(args.model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a previous output if schema checks or inference fail mid-stream.
    with tempfile.NamedTemporaryFile(prefix='.anomaly-prediction-', suffix='.partial',
                                     dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    written = 0
    try:
        for frame in pd.read_csv(source, usecols=KEYS + FEATURES, chunksize=100_000,
                                 nrows=args.max_rows):
            result = predict_observations(frame, bundle)
            result.to_csv(temporary, index=False, mode='w' if written == 0 else 'a',
                          header=written == 0, encoding='utf-8-sig' if written == 0 else 'utf-8')
            written += len(result)
        if not written:
            raise ValueError('Input contains no sensor observations')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved {written:,} sensor records to {destination}')
    print('Scores use the frozen March selection; labels are not required at inference.')


if __name__ == '__main__':
    main()
