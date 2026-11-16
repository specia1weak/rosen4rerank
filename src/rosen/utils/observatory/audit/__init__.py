"""Post-training audits built on ObservationPort traces and interventions.

Import concrete workflows, metrics and reporters from their defining modules so
lightweight trainer integrations do not eagerly load the full audit stack.
"""
