# python test_schema.py
"""

python -m analysis.test_schema

"""

import sys
import os


from analysis.analysis_schema_manager import load_analysis_schema

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    schema = load_analysis_schema()
    print("✅ YAML şeması uyumlu!")
    print(f"📊 {len(schema.modules)} modül yüklendi")
except Exception as e:
    print(f"❌ Hata: {e}")
    
