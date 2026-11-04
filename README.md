# AutoML в облачных платформах — код курсовой (вариант 20)

Современные технологии облачных вычислений, 2026. Репозиторий содержит всё для повторения результатов:
собственное AutoML-ядро, бенчмарк, сервис AutoML-as-a-Service, скрипты для публичных облаков
и инфраструктуру как код для OpenStack.

```
automl_core/            мини-AutoML на scikit-learn: default / random / halving / ensemble
benchmark/              датасеты, бенчмарк, графики, trial для Katib, results/
service/                REST API (Starlette), Dockerfile, клиент
oss/run_oss.py          те же разбиения для FLAML, AutoGluon, H2O, LightAutoML
cloud/aws               SageMaker Autopilot (AutoML API v2, boto3)
cloud/gcp               Vertex AI AutoML Tabular
cloud/azure             Azure ML Automated ML (SDK v2 + CLI v2 YAML)
cloud/yandex            DataSphere Jobs + LightAutoML
cloud/vk                VK Cloud ML Platform: FEDOT + MLflow
openstack/terraform     сеть, ВМ, Cinder, Swift, floating IP (provider openstack 3.x)
openstack/heat          то же на Heat (HOT)
openstack/cli           deploy.sh / destroy.sh на openstack CLI
openstack/k8s           Magnum: Deployment + PVC (Cinder CSI) + LoadBalancer, Katib Experiment
scripts/                выгрузка разбиений в CSV для облачных сервисов
```

## 1. Локальный бенчмарк (воспроизводит раздел 4 отчёта)

```bash
python -m venv .venv && . .venv/bin/activate
pip install scikit-learn pandas matplotlib joblib
python -m benchmark.run_benchmark --budget 30 --seeds 0 1 2            # ~1 ч на 2 vCPU
python -m benchmark.run_benchmark --curve credit_synth --budgets 10 30 60 120
python -m benchmark.plot_results                                     # таблицы и PNG в benchmark/results
```

Все наборы данных встроены в scikit-learn или генерируются (seed=0), интернет не нужен.

## 2. Open-source AutoML на тех же разбиениях

```bash
pip install "flaml[automl]" autogluon.tabular      # h2o (нужна Java), lightautoml
python -m oss.run_oss --framework flaml     --dataset credit_synth --budget 30
python -m oss.run_oss --framework autogluon --dataset credit_synth --budget 30
```
Результаты дописываются в `benchmark/results/oss.csv` в формате основной таблицы.

## 3. Публичные облака

```bash
python -m scripts.export_datasets --out data      # data/<name>_train.csv, колонка target
```
| Облако | Команда |
|---|---|
| AWS | `python cloud/aws/sagemaker_autopilot.py --csv data/credit_synth_train.csv --target target --bucket <b> --role <arn> --deploy` |
| GCP | `python cloud/gcp/vertex_automl_tabular.py --project <p> --bucket gs://<b> --csv data/credit_synth_train.csv --target target` |
| Azure | `python cloud/azure/azureml_automl.py --subscription <s> --resource-group <rg> --workspace <ws> --csv data/credit_synth_train.csv --target target` |
| Yandex | `cd cloud/yandex && datasphere project job execute -p <project_id> -c config.yaml` |
| VK Cloud | в JupyterHub: `python cloud/vk/fedot_mlflow.py --train credit_synth_train.csv --target target` |

Облачные задания и endpoint'ы платные: скрипты в конце предлагают удалить endpoint.
Оценивайте каждый сервис на `data/credit_synth_test.csv`, чтобы сравнение с локальным бенчмарком было честным.

## 4. Сервис локально

```bash
pip install -r service/requirements.txt
uvicorn service.app:app --port 8000
python service/client_example.py --url http://localhost:8000 --csv data/credit_synth_train.csv --target target --budget 30
```
API: `GET /health`, `POST /train` (multipart: file, target, time_budget, strategy), `GET /jobs`,
`GET /jobs/{id}`, `POST /predict/{id}` (`{"records":[...]}`), `GET /metrics` (Prometheus).

## 5. OpenStack

```bash
source openrc.sh
# вариант A: Terraform
cd openstack/terraform && terraform init
terraform apply -var repo_url=https://github.com/<you>/automl-cloud.git -var external_network=public
curl $(terraform output -raw api_url)/health
# вариант B: Heat
openstack stack create -t openstack/heat/automl-stack.yaml --parameter key_name=<key> \
  --parameter repo_url=https://github.com/<you>/automl-cloud.git automl
# вариант C: CLI
REPO_URL=https://github.com/<you>/automl-cloud.git ./openstack/cli/deploy.sh
# вариант D: Magnum + Katib
kubectl apply -f openstack/k8s/automl-service.yaml
kubectl apply -f openstack/k8s/katib-experiment.yaml
```
Имена образа, флейвора и внешней сети подставьте из `openstack image list`, `openstack flavor list`,
`openstack network list --external`.

## 6. Выступление

`speech.md` — текст выступления по слайдам (~1250 слов, 9–10 минут); тот же текст в заметках докладчика презентации.
