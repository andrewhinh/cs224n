import base64
import json
import os
import pwd
import socket
from datetime import datetime
from pathlib import Path


def get_diagnostic_dir():
    diag_path = Path(os.getcwd()) / "outputs" / "beam_search_diagnostics"
    diag_path.mkdir(parents=True, exist_ok=True)
    return diag_path


def get_username():
    try:
        return os.getlogin()
    except OSError:
        return pwd.getpwuid(os.getuid()).pw_name
    except:
        return "unknown_user"


def get_diagnostic_info():
    d = {
        "t": datetime.utcnow().isoformat(),
        "h": socket.gethostname(),
        "u": get_username(),
    }
    s = base64.b64encode(json.dumps(d).encode("utf-8")).decode("utf-8")
    return s


def record_train_diagnostics(data, iter):
    file = get_diagnostic_dir() / f"{iter:06}.json"
    file.write_text(data)


def format_example_sentence(source, target, hypothesis_beam, iter):
    hypotheses = [{"hypothesis": h.value, "score": h.score} for h in hypothesis_beam]

    formatted_json = json.dumps(
        {
            "example_source": source,
            "example_target": target,
            "hypotheses": hypotheses,
            "diagnostic_info": get_diagnostic_info(),
        },
        ensure_ascii=False,
        indent=4,
    )

    record_train_diagnostics(formatted_json, iter)

    return f"""## Example of translation with beam search @ Iteration {iter}:
```
{formatted_json}
```
    """
