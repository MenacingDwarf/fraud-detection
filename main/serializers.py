from rest_framework import serializers
from .models import Transaction


class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = ("identifier", "step", "type", "amount", "oldBalanceDest", "newBalanceDest",
                  "oldBalanceOrig", "newBalanceOrig", "errorBalanceDest", "errorBalanceOrig")
