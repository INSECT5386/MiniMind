
"""
def main():
    print("MiniMind 패키지 실행 - 테스트 시작!")
    
    # 여기서 간단히 NeuralGenerator 테스트 예시 실행
    from .neural import NeuralGenerator
    from .sampling import Sampler

    sampler = Sampler(method='temperature', temperature=0.8)
    
    # 더미 데이터 (토큰 인덱스 배열) 예시
    import numpy as np
    vocab_size = 100
    X_dummy = np.random.randint(0, vocab_size-1, size=(50, 10))  # 50샘플, 길이10 시퀀스
    y_dummy = np.zeros((50, vocab_size))
    for i in range(50):
        y_dummy[i, np.random.randint(0, vocab_size)] = 1.0  # 랜덤 원핫 출력
    
    ng = NeuralGenerator(vocab_size=vocab_size, epochs=3, verbose=True, sampler=sampler)
    ng.fit(X_dummy, y_dummy)
    
    prompt = np.array([1, 2, 3])  # 시작 토큰 시퀀스 예시
    generated_seq = ng.generate(prompt, max_tokens=10)
    print("생성된 시퀀스:", generated_seq)

if __name__ == "__main__":
    main()

"""

# MiniMind/__main__.py
"""
from .sap import SAPGenerator
from .tokenizer import SimpleTokenizer
def main():
    print("MiniMind SAPGenerator 테스트 시작!")

    # 간단한 데이터 샘플 (입력-출력 쌍)
    pairs = [
        ("안녕하세요", "안녕하세요"),
        ("오늘 날씨 어때?", "날씨가 좋아요"),
        ("밥 먹었어?", "네, 잘 먹었어요"),
        ("영화 볼래?", "좋아요 같이 보자"),
        ("잘 자요", "안녕히 주무세요"),
    ]

    # SAPGenerator 인스턴스 생성 및 학습
    tokenizer = SimpleTokenizer()
    sap_gen = SAPGenerator(tokenizer=tokenizer)
    sap_gen.fit(pairs)

    # 생성 테스트
    prompt = "오늘"
    print(f"입력: {prompt}")
    generated = sap_gen.generate(prompt, max_tokens=10)
    print(f"생성 결과: {generated}")

if __name__ == "__main__":
    main()
"""

"""
from .gpm import GPMGenerator
from .sampling import Sampler
from .tokenizer import SimpleTokenizer

def main():
    print("MiniMind GPMGenerator 테스트 시작!")

    # 샘플 데이터
    pairs = [
        ("안녕하세요", "안녕하세요 반갑습니다"),
        ("오늘 날씨 어때?", "오늘은 맑고 따뜻해요"),
        ("뭐 먹을래?", "저는 김치찌개 좋아해요"),
    ]

    # 샘플러 생성 (top-k 예시)
    sampler = Sampler(method='top_k', k=3)
    tokenizer = SimpleTokenizer()

    # 생성기 초기화 시 sampler 연결
    gpm = GPMGenerator(sampler=sampler, tokenizer=tokenizer)
    gpm.fit(pairs)

    # 생성 테스트
    prompt = "안녕하세요"
    generated_text = gpm.generate(prompt, max_tokens=10)

    print("입력 프롬프트:", prompt)
    print("생성된 텍스트:", generated_text)

if __name__ == "__main__":
    main()
"""


"""
# test_sampling.py

import numpy as np
from .sampling import top_k_sampling, top_p_sampling, temperature_sampling, Sampler

def dummy_probs(size=100):
    probs = np.random.rand(size)
    return probs / probs.sum()

def test_sampling_functions():
    probs = dummy_probs()

    print("top_k_sampling:", top_k_sampling(probs, k=5))
    print("top_p_sampling:", top_p_sampling(probs, p=0.8))
    print("temperature_sampling (temp=0.5):", temperature_sampling(probs, temperature=0.5))
    print("temperature_sampling (temp=2.0):", temperature_sampling(probs, temperature=2.0))

def test_sampler_class():
    probs = dummy_probs()
    sampler = Sampler(method='top_p', p=0.9)
    print("Sampler top_p:", sampler.sample(probs))

    sampler.method = 'top_k'
    sampler.k = 3
    print("Sampler top_k:", sampler.sample(probs))

    sampler.method = 'temperature'
    sampler.temperature = 0.7
    print("Sampler temperature:", sampler.sample(probs))

if __name__ == "__main__":
    test_sampling_functions()
    test_sampler_class()
"""
"""
from .tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

text = "Hello, 안녕하세요! Let's test the tokenizer 123."
tokens = tokenizer.tokenize(text)
print("토큰:", tokens)

reconstructed = tokenizer.detokenize(tokens)
print("복원된 문장:", reconstructed)
"""

"""
import os
import numpy as np
from .utils import set_seed, save_json, load_json, save_model_weights, load_model_weights, simple_logger


if __name__ == "__main__":
    # 테스트 함수들

    def test_set_seed():
        set_seed(123)
        a = np.random.rand(3)
        set_seed(123)
        b = np.random.rand(3)
        assert np.allclose(a, b), "set_seed 실패!"
        print("set_seed 테스트 통과!")

    def test_save_load_json():
        data = {'name': 'MiniMind', 'version': 1.0}
        filepath = 'test.json'
        save_json(data, filepath)
        loaded = load_json(filepath)
        assert data == loaded, "JSON 저장/로드 실패!"
        os.remove(filepath)
        print("save_json & load_json 테스트 통과!")

    def test_save_load_weights_multi_format():
        weights = {
            'W1': np.array([1, 2, 3]),
            'b1': np.array([0.1, 0.2, 0.3])
        }
        for fmt in ['npz', 'joblib', 'json']:
            filepath = f"weights_test.{fmt}"
            save_model_weights(weights, filepath, format=fmt)
            loaded = load_model_weights(filepath, format=fmt)
            for k in weights:
                assert np.allclose(weights[k], loaded[k]), f"{fmt} {k} 가중치 저장/로드 실패!"
            os.remove(filepath)
        print("멀티 포맷 가중치 저장/로드 테스트 통과!")

    def test_logger():
        simple_logger("테스트 로그 메시지")

    # 실행 테스트 모음
    test_set_seed()
    test_save_load_json()
    test_save_load_weights_multi_format()
    test_logger()

"""