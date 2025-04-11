python scripts/dump_bin.py dump_all \
  --csv_path data/binance/1m \
  --qlib_dir ~/.qlib/qlib_data/binance \
  --date_field_name datetime \
  --include_fields open,close,high,low,volume \
  --freq 1min
