# Fix Notes

The previous POC called Chronos with keyword arguments:

```python
pipeline.predict(context=context, prediction_length=...)
```

Some installed `chronos-forecasting` versions expose `predict()` without accepting `context=` as a keyword, causing:

`TypeError: ChronosPipeline.predict() got an unexpected keyword argument 'context'`

The fixed POC now follows the official Chronos-T5 usage:

```python
pipeline.predict(context, prediction_length)
```

No GPR, residual-learning, evaluation, or thesis logic was changed.
