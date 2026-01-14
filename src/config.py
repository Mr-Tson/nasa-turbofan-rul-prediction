from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent / 'outputs'

# Preprocessing configuration
RUL_MAX = 130
DATASET = 'FD001'

index_names = ['engine', 'cycle']
setting_names = ['throttle resolver angle', 'altitude', 'mach number']
sensor_names=[ "(Fan inlet total temperature) (◦R)",
"(LPC outlet total temperature) (◦R)",
"(HPC outlet total temperature) (◦R)",
"(LPT outlet total temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct total pressure) (psia)",
"(HPC outlet total pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Demanded fan speed) (rpm)",
"(Demanded corrected fan speed) (rpm)",
"(High-pressure turbines coolant bleed) (lbm/s)",
"(Low-pressure turbines coolant bleed) (lbm/s)" ]
col_names = index_names + setting_names + sensor_names
