import loguru


def bind_logger(component: object | str, context: str | None = None) -> loguru.Logger:
    label: str
    match component:
        case str():
            label = component
        case _:
            label = component.__class__.__name__
    if context is not None:
        label += f"({context})"
    return loguru.logger.bind(component=label)
