# pipeline

스크립트 81개(`.py` 39 + SLURM 러너 `.sh` 42). 역할 지도는 [`../docs/SCRIPTS.md`](../docs/SCRIPTS.md)에 있다.

- **과정 A**(feature 모으기): 01~27, 99
- **과정 B**(모델·선별·검증): 28~37
- **다음 사이클**: 38

스크립트가 `Path(__file__).with_name(...)`으로 **같은 폴더의 형제 파일을 import**하기 때문에, 하위 폴더로 나누지 않았다.

서버 경로는 `<DATA_ROOT>` 등 자리표시자로 바꿨다. 실행하려면 자기 환경의 경로로 되돌려야 한다
(규칙은 [`../docs/ANONYMIZATION.md`](../docs/ANONYMIZATION.md)). 치환 전 원본 해시는 `ORIGINAL_SHA256SUMS.txt`에 있다.
