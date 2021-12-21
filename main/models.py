from django.db import models
from django.contrib.auth.models import User


class Transaction(models.Model):
    ownerBank = models.ForeignKey(User, models.CASCADE, related_name="transactions")
    identifier = models.IntegerField(primary_key=True)
    step = models.IntegerField()
    type = models.IntegerField()
    amount = models.FloatField()
    oldBalanceDest = models.FloatField(null=True, blank=True)
    newBalanceDest = models.FloatField(null=True, blank=True)
    oldBalanceOrig = models.FloatField(null=True, blank=True)
    newBalanceOrig = models.FloatField(null=True, blank=True)
    errorBalanceDest = models.FloatField(null=True, blank=True)
    errorBalanceOrig = models.FloatField(null=True, blank=True)
    trained = models.BooleanField(default=False)
    prediction = models.BooleanField(default=False)
