import abc
import os
import json
from .tokenizer import SimpleTokenizer

class BaseGenerator(abc.ABC):
    """
    MiniMind 생성기들의 공통 추상 기반 클래스
    fit(), generate()는 반드시 서브클래스에서 구현해야 함.
    save(), load()는 기본 JSON 직렬화 지원. 필요하면 오버라이드 가능.
    """

    def __init__(self, sampler=None, tokenizer=ModuleNotFoundError):
        self.is_fitted = False
        self.sampler = sampler
        self.model_state = {}
        self.tokenizer = tokenizer or SimpleTokenizer()

    @abc.abstractmethod
    def fit(self, pairs):
        """
        (input_text, output_text) 쌍 리스트로 모델 학습.
        서브클래스에서 반드시 구현할 것.
        """
        raise NotImplementedError("fit()는 서브클래스에서 구현하세요.")

    @abc.abstractmethod
    def generate(self, prompt, max_tokens=20, **kwargs):
        """
        prompt 입력받아 텍스트 생성.
        서브클래스에서 반드시 구현할 것.
        """
        raise NotImplementedError("generate()는 서브클래스에서 구현하세요.")

    def save(self, path):
        """
        모델 상태를 JSON 파일로 저장.
        """
        state = self._get_state()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        JSON 파일에서 모델 상태 불러오기.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' 파일이 존재하지 않습니다.")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._set_state(state)
        self.is_fitted = True

    def _get_state(self):
        """
        저장할 상태를 딕셔너리 형태로 반환.
        서브클래스에서 필요한 데이터 포함하도록 오버라이드 가능.
        """
        return self.model_state

    def _set_state(self, state):
        """
        불러온 상태 딕셔너리를 내부 변수에 할당.
        서브클래스에서 필요한 데이터 처리하도록 오버라이드 가능.
        """
        self.model_state = state
