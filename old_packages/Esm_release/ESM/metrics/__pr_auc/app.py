import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("pr_auc")
launch_gradio_widget(module)
