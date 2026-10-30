solar_sites() {
  cat <<'EOF'
alabama5min.csv alabama5min alabama 32 -87 America/Chicago 137
california5min.csv california5min california 35 -117 America/Los_Angeles 405
newyork5min.csv newyork5min newyork 43 -75 America/New_York 129
ohio5min.csv ohio5min ohio 40 -83 America/New_York 95
EOF
}

solar_transfer_sites() {
  cat <<'EOF'
newyork5min.csv newyork 43 -75 America/New_York 129
ohio5min.csv ohio 40 -83 America/New_York 95
EOF
}

SOLAR_PRED_LENS="24 48 96 144 288"
SOLAR_RANDOM_SEEDS="2024"
SOLAR_SEQ_LEN=144
SOLAR_ROOT=./dataset/
SOLAR_DATA=custom
