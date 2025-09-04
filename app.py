import requests
from flask import Flask, jsonify
from collections import Counter, defaultdict
import math
import os
import logging
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# URL API gốc để lấy dữ liệu
SOURCE_API_URL = "https://ahihidonguoccut-2b5i.onrender.com/mohobomaycai"

# Biến toàn cục lưu trữ lịch sử các phiên và dự đoán
historical_data = []
prediction_history = []

# --- FEATURE ENGINEERING & ADVANCED AI MODELS (NÂNG CẤP FULL) ---

def feature_engineering(sessions: list, pattern: str):
    """
    Tạo ra một bộ đặc trưng (features) phong phú từ dữ liệu lịch sử.
    Đây là bước nền tảng cho các mô hình AI nâng cao.
    """
    features = {}
    if not pattern:
        return features

    # --- Feature cơ bản ---
    # Độ dài cầu hiện tại
    last_char = pattern[-1]
    run_len = 0
    for char in reversed(pattern):
        if char == last_char: run_len += 1
        else: break
    features['run_len'] = run_len

    # Số phiên kể từ lần đổi cầu cuối
    time_since_change = 0
    for char in reversed(pattern[1:]):
        if char == pattern[-1]: time_since_change += 1
        else: break
    features['time_since_change'] = time_since_change + 1

    # --- Feature thống kê theo cửa sổ (window) ---
    for k in [5, 10, 20, 50]:
        if len(pattern) >= k:
            window = pattern[-k:]
            # Tỷ lệ Tài
            features[f'ratio_T_in_window_{k}'] = window.count('T') / k
            # Entropy - đo độ hỗn loạn/ngẫu nhiên của chuỗi
            counts = Counter(window)
            probs = [count / k for count in counts.values()]
            features[f'entropy_{k}'] = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            features[f'ratio_T_in_window_{k}'] = None
            features[f'entropy_{k}'] = None
    
    # --- Feature xu hướng điểm xúc xắc ---
    if len(sessions) >= 10:
        scores = [s['Tong'] for s in sessions[-10:]]
        avg1 = np.mean(scores[:5])
        avg2 = np.mean(scores[5:])
        features['dice_sum_trend'] = avg2 - avg1 # > 0: tăng, < 0: giảm
    else:
        features['dice_sum_trend'] = 0

    return features

def model_entropy_based(pattern: str, features: dict):
    """
    Phân tích "độ đẹp" của cầu dựa trên Entropy.
    - Entropy thấp: Chuỗi có quy luật, cầu rõ ràng -> Đi theo xu hướng.
    - Entropy cao: Chuỗi ngẫu nhiên, loạn cầu -> Dự đoán cân bằng/bẻ cầu.
    """
    entropy = features.get('entropy_20')
    if entropy is None:
        return None, 0.0, "Không đủ dữ liệu để tính entropy."

    if entropy < 0.75: # Cầu đang có quy luật rất mạnh
        pred = pattern[-1]
        conf = (1 - entropy) * 0.9 # Entropy càng thấp, conf càng cao
        return pred, conf, f"Entropy thấp ({entropy:.2f}), cầu đang có quy luật, theo {pred}."
    
    if entropy > 0.98: # Cầu rất loạn, khả năng cao sẽ cân bằng lại
        ratio_t = features.get('ratio_T_in_window_20', 0.5)
        pred = 'X' if ratio_t > 0.55 else 'T' # Bẻ về phía ngược lại
        conf = (entropy - 0.95) * 2 # Entropy càng cao, conf bẻ càng cao
        return pred, conf, f"Entropy cao ({entropy:.2f}), loạn cầu, dự đoán cân bằng {pred}."
        
    return None, 0.0, "Entropy ở mức trung bình."

def model_block_pattern(pattern: str):
    """
    Phát hiện các mẫu cầu lặp lại theo khối (block detection).
    Ví dụ: (TTX)(TTX), (TXX)(TXX)
    """
    if len(pattern) < 6:
        return None, 0.0, "Không đủ dữ liệu."
    
    for block_size in [2, 3, 4]:
        if len(pattern) >= block_size * 2:
            block1 = pattern[-block_size*2 : -block_size]
            block2 = pattern[-block_size:]
            if block1 == block2:
                pred = block1[0] # Dự đoán ký tự đầu tiên của block tiếp theo
                return pred, 0.9, f"Phát hiện cầu lặp khối {block_size} ({block1})."
    
    return None, 0.0, "Không có cầu lặp khối."

def model_feature_synthesis(pattern: str, features: dict):
    """
    AI Meta-Model: Tổng hợp các features để đưa ra quyết định cấp cao.
    Đây là model thông minh nhất, chỉ đưa ra dự đoán khi tín hiệu hội tụ mạnh.
    """
    # QUY TẮC 1: BÁM CẦU BỆT MẠNH (STRONG RUN FOLLOWING)
    # Điều kiện: Cầu dài >= 4, entropy thấp, xu hướng điểm ủng hộ
    if features.get('run_len', 0) >= 4 and features.get('entropy_10', 1) < 0.8:
        pred = pattern[-1]
        trend = features.get('dice_sum_trend', 0)
        if (pred == 'T' and trend >= 0) or (pred == 'X' and trend <= 0):
             return pred, 0.95, f"Hội tụ tín hiệu bám cầu {pred} (dài {features['run_len']}, entropy thấp, trend tốt)."

    # QUY TẮC 2: BẺ CẦU KHI CÓ DẤU HIỆU QUÁ TẢI (OVERLOAD REVERSAL)
    # Điều kiện: Tỷ lệ T/X trong 20 phiên gần nhất quá cao (>75%)
    ratio_t_20 = features.get('ratio_T_in_window_20')
    if ratio_t_20 is not None:
        if ratio_t_20 > 0.75:
            return 'X', (ratio_t_20 - 0.7) * 2, f"Tài ra quá nhiều ({ratio_t_20:.2f}), ưu tiên bẻ Xỉu."
        if ratio_t_20 < 0.25:
            return 'T', ((1-ratio_t_20) - 0.7) * 2, f"Xỉu ra quá nhiều ({1-ratio_t_20:.2f}), ưu tiên bẻ Tài."

    # QUY TẮC 3: CẦU 1-1 TRONG GIAI ĐOẠN LOẠN CẦU
    # Điều kiện: Cầu 1-1 và entropy cao
    if pattern[-4:] in ["TXTX", "XTXT"] and features.get('entropy_10', 0) > 0.95:
        pred = 'T' if pattern[-1] == 'X' else 'X'
        return pred, 0.88, f"Cầu 1-1 trong giai đoạn loạn cầu, tín hiệu mạnh."

    return None, 0.0, "Không có tín hiệu hội tụ đủ mạnh."

# --- CÁC MODEL CŨ VẪN GIỮ LẠI ĐỂ THAM KHẢO ---
# (Toàn bộ các hàm model cũ từ model_probability đến model_10_can_kiet được giữ nguyên ở đây)
def model_probability(pattern: str, window_sizes=[10, 20, 50]):
    """Thống kê xác suất với rolling window"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < min(window_sizes): return 'T', 0.5, f"Chưa đủ {min(window_sizes)} phiên, mặc định Tài."
    scores = []
    for window in window_sizes:
        if len(pattern) >= window:
            recent = pattern[-window:]
            t_ratio = recent.count('T') / window
            if t_ratio > 0.7: scores.append(('X', t_ratio - 0.5, f"Rolling {window}: Tài cao ({t_ratio:.2f}), ưu tiên Xỉu."))
            elif t_ratio < 0.3: scores.append(('T', 0.5 - t_ratio, f"Rolling {window}: Xỉu cao ({t_ratio:.2f}), ưu tiên Tài."))
    if not scores: return None, 0.0, "Không có tín hiệu."
    vote_count = Counter([s[0] for s in scores])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(s[1] for s in scores if s[0] == pred) / len([s for s in scores if s[0] == pred])
    reason = " | ".join(s[2] for s in scores if s[0] == pred)
    return pred, conf, reason

def model_markov(pattern: str, order=1):
    """Markov Chain: Xác suất chuyển trạng thái (order=1)"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5: return 'T', 0.5, f"Chưa đủ 5 phiên, mặc định Tài."
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(pattern) - order):
        state = pattern[i:i+order]
        next_state = pattern[i+order]
        transitions[state][next_state] += 1
    current_state = pattern[-order:]
    if current_state not in transitions: return None, 0.0, "Không có transition."
    total = sum(transitions[current_state].values())
    probs = {k: v / total for k, v in transitions[current_state].items()}
    if not probs: return None, 0.0, "Không có prob."
    pred = max(probs, key=probs.get)
    conf = probs[pred] * min(1.0, len(pattern) / 50)
    reason = f"Markov (order {order}): Từ '{current_state}' -> '{pred}' với prob {probs[pred]:.2f}."
    return pred, conf, reason

def model_ngram(pattern: str, n_range=(3,6)):
    """N-gram Pattern Matching"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < max(n_range) + 1: return 'T', 0.5, f"Chưa đủ {max(n_range)+1} phiên, mặc định Tài."
    scores = []
    for n in range(n_range[0], n_range[1]+1):
        if len(pattern) < n: continue
        last_n = pattern[-n:]
        matches = [i for i in range(len(pattern) - n) if pattern[i:i+n] == last_n]
        if len(matches) < 2: continue
        next_chars = [pattern[i+n] for i in matches if i+n < len(pattern)]
        count = Counter(next_chars)
        if not count: continue
        pred, freq = count.most_common(1)[0]
        conf = (freq / len(next_chars)) * min(1.0, len(next_chars) / 5)
        scores.append((pred, conf, f"N-gram {n}: Sau '{last_n}' ra '{pred}' ({freq}/{len(next_chars)})."))
    if not scores: return None, 0.0, "Không có match."
    vote_count = Counter([s[0] for s in scores])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(s[1] for s in scores if s[0] == pred) / len([s for s in scores if s[0] == pred])
    reason = " | ".join(s[2] for s in scores if s[0] == pred)
    return pred, conf, reason

def model_heuristic(pattern: str, sessions: list):
    """Heuristic: Kết hợp pattern cơ bản"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    sub_models = [
        model_1_bet_va_1_1(pattern),
        model_2_cau_nhip(pattern),
        model_4_cau_phuc_tap(pattern),
    ]
    valid = [m for m in sub_models if m[0] is not None]
    if not valid: return None, 0.0, "Không có tín hiệu."
    vote_count = Counter([m[0] for m in valid])
    pred = vote_count.most_common(1)[0][0]
    conf = sum(m[1] for m in valid if m[0] == pred) / len([m for m in valid if m[0] == pred])
    reason = "Heuristic: Kết hợp pattern cơ bản."
    return pred, conf, reason

def model_1_bet_va_1_1(pattern: str):
    """Phân tích cầu bệt và 1-1"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 3: return 'T', 0.5, "Chưa đủ 3 phiên, mặc định Tài."
    if pattern.endswith('TTT'):
        b_len = len(pattern) - len(pattern.rstrip('T'))
        conf = min(1.0, 0.6 + (b_len - 3) * 0.1)
        return 'T', conf, f"Bệt Tài {b_len} phiên."
    if pattern.endswith('XXX'):
        b_len = len(pattern) - len(pattern.rstrip('X'))
        conf = min(1.0, 0.6 + (b_len - 3) * 0.1)
        return 'X', conf, f"Bệt Xỉu {b_len} phiên."
    if pattern[-3:] in ["TXT", "XTX"]:
        return ('T' if pattern[-1] == 'X' else 'X'), 0.8, "Cầu 1-1."
    return None, 0.0, "Không tín hiệu."

def model_2_cau_nhip(pattern: str):
    """Phân tích cầu nhịp 1-2, 2-1, 2-2"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 4: return 'T', 0.5, "Chưa đủ 4 phiên, mặc định Tài."
    if pattern[-4:] in ["XXTT", "TTXX"]: return ('X' if pattern[-4:] == "XXTT" else 'T'), 0.85, "Cầu 2-2."
    if pattern[-3:] in ["TXX", "XTT"]: return ('T' if pattern[-3:] == "TXX" else 'X'), 0.75, "Cầu 1-2."
    if pattern[-3:] in ["TTX", "XXT"]: return ('X' if pattern[-3:] == "TTX" else 'T'), 0.75, "Cầu 2-1."
    return None, 0.0, "Không tín hiệu."

def model_3_thong_ke(pattern: str):
    """Phân tích thống kê mẫu lặp lại"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 10: return 'T', 0.5, "Chưa đủ 10 phiên, mặc định Tài."
    last_3 = pattern[-3:]
    starts = [i for i in range(len(pattern) - 3) if pattern[i:i+3] == last_3]
    if len(starts) < 2: return None, 0.0, f"Mẫu '{last_3}' ít."
    nexts = [pattern[i+3] for i in starts if i+3 < len(pattern)]
    count = Counter(nexts)
    if not count: return None, 0.0, "Không dữ liệu."
    pred, num = count.most_common(1)[0]
    conf = (num / len(nexts)) * min(1.0, len(nexts) / 4)
    return pred, conf, f"Sau '{last_3}' ra '{pred}' ({num}/{len(nexts)})."

def model_4_cau_phuc_tap(pattern: str):
    """Phân tích cầu 3-1, 1-3"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5: return 'T', 0.5, "Chưa đủ 5 phiên, mặc định Tài."
    if pattern[-4:] == "TTTX": return 'X', 0.7, "Cầu 3-1."
    if pattern[-4:] == "XXXT": return 'T', 0.7, "Cầu 1-3."
    return None, 0.0, "Không tín hiệu."

def model_5_cau_4_1(pattern: str):
    """Phân tích cầu 4-1, 1-4"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6: return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] == "TTTTX": return 'X', 0.65, "Cầu 4-1."
    if pattern[-5:] == "XXXXT": return 'T', 0.65, "Cầu 1-4."
    return None, 0.0, "Không tín hiệu."

def model_6_cau_2_3(pattern: str):
    """Phân tích cầu 2-3, 3-2"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6: return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] == "TTXXX": return 'T', 0.7, "Cầu 2-3."
    if pattern[-5:] == "XXTTT": return 'X', 0.7, "Cầu 3-2."
    return None, 0.0, "Không tín hiệu."

def model_7_score_trend(sessions: list):
    """Phân tích xu hướng điểm số"""
    if not sessions: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(sessions) < 8: return 'T', 0.5, "Chưa đủ 8 phiên, mặc định Tài."
    scores = [s['Tong'] for s in sessions[-8:]]
    half = len(scores) // 2
    avg1 = sum(scores[:half]) / half
    avg2 = sum(scores[half:]) / half
    diff = avg2 - avg1
    conf = min(1.0, abs(diff) / 3.0)
    if diff > 0.75: return 'T', conf, f"Điểm tăng ({avg2:.1f} > {avg1:.1f})."
    if diff < -0.75: return 'X', conf, f"Điểm giảm ({avg2:.1f} < {avg1:.1f})."
    return None, 0.0, "Xu hướng không rõ."

def model_8_cau_1_2_1(pattern: str):
    """Phân tích cầu 1-2-1"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 5: return 'T', 0.5, "Chưa đủ 5 phiên, mặc định Tài."
    if pattern[-4:] in ["TXTT", "XTXX"]: return ('X' if pattern[-4:] == "TXTT" else 'T'), 0.75, "Cầu 1-2-1."
    return None, 0.0, "Không tín hiệu."

def model_9_cau_2_1_2(pattern: str):
    """Phân tích cầu 2-1-2"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 6: return 'T', 0.5, "Chưa đủ 6 phiên, mặc định Tài."
    if pattern[-5:] in ["TTXTX", "XXTXT"]: return ('T' if pattern[-5:] == "XXTXT" else 'X'), 0.75, "Cầu 2-1-2."
    return None, 0.0, "Không tín hiệu."

def model_10_can_kiet(pattern: str):
    """Phân tích cân bằng dài hạn"""
    if not pattern: return 'T', 0.5, "Không có dữ liệu, mặc định Tài."
    if len(pattern) < 20: return 'T', 0.5, "Chưa đủ 20 phiên, mặc định Tài."
    t_ratio = pattern[-20:].count('T') / 20
    if t_ratio >= 0.7: return 'X', (t_ratio - 0.6) * 2.5, "Tài cao, cân bằng Xỉu."
    if t_ratio <= 0.3: return 'T', (0.4 - t_ratio) * 2.5, "Xỉu cao, cân bằng Tài."
    return None, 0.0, "Cân bằng."
# --- API ROUTING ---

@app.route('/predict_vip', methods=['GET'])
def predict_vip():
    global historical_data, prediction_history
    try:
        response = requests.get(SOURCE_API_URL, timeout=10)
        response.raise_for_status()
        latest_data = response.json()
        if not historical_data or historical_data[-1]['Phien'] != latest_data['Phien']:
            historical_data.append(latest_data)
            logger.info(f"Đã thêm phiên mới: {latest_data['Phien']}")
    except requests.RequestException as e:
        logger.error(f"Lỗi API nguồn: {str(e)}")
        return jsonify({"error": f"Lỗi API nguồn: {str(e)}"}), 500

    if len(historical_data) < 10:
        return jsonify({"current_session": 0, "next_session": 1, "du_doan": "Tài", "confidence": "50%", "meta": "Chưa đủ dữ liệu, mặc định Tài.", "id": "Tele@HoVanThien_Pro"}), 200

    recent_sessions = historical_data[-200:]
    recent_pattern = "".join(['T' if s['Ket_qua'] == 'Tài' else 'X' for s in recent_sessions])
    
    # BƯỚC 1: TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)
    features = feature_engineering(recent_sessions, recent_pattern)

    # BƯỚC 2: CẤU HÌNH ENSEMBLE MODEL THÔNG MINH
    models_config = [
        # --- Model AI Nâng cao có trọng số cao nhất ---
        {"name": "🤖 AI Meta-Model Synthesis", "func": model_feature_synthesis, "weight": 2.5, "args": (recent_pattern, features)},
        {"name": "🔍 Entropy & Chaos Analysis", "func": model_entropy_based, "weight": 1.8, "args": (recent_pattern, features)},
        {"name": "🧱 Block Pattern Detection", "func": model_block_pattern, "weight": 1.7, "args": (recent_pattern,)},

        # --- Các model thống kê và quy tắc cơ bản ---
        {"name": "Bệt/1-1", "func": model_1_bet_va_1_1, "weight": 1.5, "args": (recent_pattern,)},
        {"name": "Nhịp 1-2/2-1/2-2", "func": model_2_cau_nhip, "weight": 1.5, "args": (recent_pattern,)},
        {"name": "Markov Chain", "func": model_markov, "weight": 1.4, "args": (recent_pattern,)},
        {"name": "N-gram Matching", "func": model_ngram, "weight": 1.2, "args": (recent_pattern,)},
        {"name": "Probability Window", "func": model_probability, "weight": 1.0, "args": (recent_pattern,)},
        {"name": "Trend Điểm Số", "func": model_7_score_trend, "weight": 1.0, "args": (recent_sessions,)},
        {"name": "Cân Bằng Dài Hạn", "func": model_10_can_kiet, "weight": 0.8, "args": (recent_pattern,)},
    ]

    score_tai = 0.0
    score_xiu = 0.0
    model_details = []

    for config in models_config:
        try:
            pred, conf, reason = config["func"](*config["args"])
            if pred:
                score = config["weight"] * conf
                if pred == 'T': score_tai += score
                elif pred == 'X': score_xiu += score
                model_details.append({"model": config["name"], "pred": pred, "conf": f"{conf:.2f}", "reason": reason})
        except Exception as e:
            logger.error(f"Lỗi model {config['name']}: {str(e)}")
            continue
    
    # BƯỚC 3: RA QUYẾT ĐỊNH CUỐI CÙNG
    final_pred = "Tài" if score_tai >= score_xiu else "Xỉu"
    total_score = score_tai + score_xiu
    conf_val = 50 + (abs(score_tai - score_xiu) / total_score * 50) if total_score > 0 else 50
    conf_str = f"{min(98, int(conf_val))}%"

    meta = ""
    # Tìm lý do từ model có trọng số cao nhất đã đưa ra dự đoán
    strongest_reason = "Tổng hợp nhiều tín hiệu."
    for detail in model_details:
        if detail['pred'] == final_pred and any(cfg['name'] == detail['model'] and cfg['weight'] > 1.5 for cfg in models_config):
            strongest_reason = detail['reason']
            meta = f"Tín hiệu mạnh từ {detail['model']}."
            break
    
    last_session = historical_data[-1]
    # ... (phần cập nhật lịch sử dự đoán giữ nguyên)

    response = {
        "current_session": last_session['Phien'],
        "dice": [last_session.get('Xuc_xac_1', 0), last_session.get('Xuc_xac_2', 0), last_session.get('Xuc_xac_3', 0)],
        "total": last_session.get('Tong', 0),
        "result": last_session.get('Ket_qua', 'N/A'),
        "next_session": last_session['Phien'] + 1,
        "du_doan": final_pred,
        "confidence": conf_str,
        "meta": meta,
        "reasoning": strongest_reason, # Thêm trường giải thích rõ hơn
        "models": model_details,
        "features": {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in features.items() if v is not None}, # Gửi kèm features đã tính
        "id": "Tele@HoVanThien_Pro"
    }
    return jsonify(response)

# ... (các route /history-predict và /health giữ nguyên) ...
@app.route('/history-predict', methods=['GET'])
def history_predict():
    return jsonify({
        "history": prediction_history,
        "total": len(prediction_history),
        "id": "@ Văn Nhật Trở Lại"
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "id": "Tele@HoVanThien_Pro"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

