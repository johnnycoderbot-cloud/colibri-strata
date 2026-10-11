"""Production native C inference adapter factory.

The existing Python control plane transports bytes. Model execution, native SDK
calls, token admission, templating and image tensor preparation live in the C
engine. The former Python adapters are offline references under experimental/.
"""


class StreamCancelled(RuntimeError):
    """A native request was cancelled before publishing its result."""


def make_adapter(manifest, model_dir, device="AUTO"):
    from native_graph import NativeGraphAdapter
    return NativeGraphAdapter(manifest, model_dir, device)
