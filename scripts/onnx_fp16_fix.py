"""Дочинить граф после onnxconverter_common.float16: у операций, где входы должны
быть одного типа, а один пришёл f32 (константы, посчитанные на лету: эмбеддинг
времени, масштаб внимания), — поставить Cast в f16. Повторяет, пока граф не
загрузится в ONNX Runtime.

    python scripts/onnx_fp16_fix.py IN.onnx OUT.onnx
"""
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference

SAME_TYPE = {"Add", "Sub", "Mul", "Div", "Pow", "Gemm", "MatMul", "Where", "Max", "Min", "Concat", "Equal", "Less",
             "Greater", "Sum", "Mean"}
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
model = onnx.load(str(src))
for rnd in range(10):
    tmp = dst.with_suffix(".typed.onnx")
    # onnx.save с внешними данными переводит модель на них на месте — сохраняется копия
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    onnx.save(copy, str(tmp), save_as_external_data=True, all_tensors_to_one_file=True, location=tmp.name + ".data")
    del copy
    shape_inference.infer_shapes_path(str(tmp), str(tmp))
    typed = onnx.load(str(tmp), load_external_data=False)
    types = {v.name: v.type.tensor_type.elem_type for v in list(typed.graph.value_info) + list(typed.graph.input)
             + list(typed.graph.output)}
    types.update({t.name: t.data_type for t in typed.graph.initializer})
    for n in typed.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    types[n.output[0]] = a.t.data_type
    g = model.graph
    fixed = 0
    for n in list(g.node):
        if n.op_type not in SAME_TYPE:
            continue
        ins = [i for i in n.input if i]
        kinds = {types.get(i) for i in ins}
        if TensorProto.FLOAT16 in kinds and TensorProto.FLOAT in kinds:
            for k, i in enumerate(n.input):
                if i and types.get(i) == TensorProto.FLOAT:
                    out = f"{n.name}/voicy_f16_{k}"
                    g.node.insert(list(g.node).index(n), helper.make_node("Cast", [i], [out], to=TensorProto.FLOAT16,
                                                                         name=f"{n.name}/voicy_cast_{k}"))
                    n.input[k] = out
                    fixed += 1
    for f in (tmp, Path(str(tmp) + ".data")):
        f.unlink(missing_ok=True)
    print(f"проход {rnd + 1}: приведений {fixed}", flush=True)
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    onnx.save(copy, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + ".data")
    del copy
    try:
        ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        print("граф загружается", flush=True)
        break
    except Exception as e:
        print("  ещё:", str(e)[:200], flush=True)
        if fixed == 0:
            sys.exit(1)
