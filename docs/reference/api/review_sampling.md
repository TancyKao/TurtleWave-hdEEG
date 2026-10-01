# Review Sampling API Reference

`turtlewave_hdEEG.review_sampling` draws a stratified random sample of one
detection scope, scores a reviewer's labels on it, and estimates precision.

- `preview_allocation` returns what `draw_review_sample` would draw, cell by
  cell, without writing anything. It is what the GUI's draw dialog shows.
- `draw_review_sample` draws (or returns) a sample and writes
  `review_sample_designs` and `review_samples`.
- `read_sample_labels` lists every stored label on a sample, flagging labels
  voided because the event is gone or its end time moved.
- `sample_progress` reports how far a reviewer has got and what to show next.
- `compute_review_precision` returns a design-weighted precision per domain and
  writes `review_precision`.
- `label_agreement` scores two raters' labels on the same events (Cohen's kappa,
  percent agreement, positive and negative agreement).
- `top_up_region` adds 12 events to one region, once.
- `wilson`, `mover_difference` and `weighted_precision` are the pure estimator
  pieces.

The verdict rule in `POOLING` is a lab convention, not a published standard.
Decisions are read from `event_reviews` by event uuid. In 4.6 that table has no
`sample_id` column, so a decision made outside the sample on a sampled event
counts toward the sample.

See [Validate a detection run](../../how-to/validate-a-detection-run.md) for the
procedure and
[Event figures and review sampling](../../explanation/event-figures-and-review-sampling.md)
for the design.

::: turtlewave_hdEEG.review_sampling
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
