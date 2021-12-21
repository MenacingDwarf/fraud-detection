from django.http import JsonResponse
from django.shortcuts import render

from django.contrib.auth import authenticate

from .models import *
from .serializers import *
from .model_training.model_training import *

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from rest_framework import authentication
from rest_framework.authtoken.models import Token


def hello(request):
    return JsonResponse({"data": "hello"})


def reg_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = User.objects.create_user(
        username=username,
        password=password
    )
    return user


class Customer(APIView):
    permission_classes = [permissions.IsAuthenticated, ]
    authentication_classes = (authentication.TokenAuthentication,)

    def get(self, request):
        user = authenticate(username=request.POST['username'], password=request.POST['password'])
        if user and user.is_active is True and user.username == "admin":
            token = Token.objects.get(user=user)
            if token:
                return Response({"status": 200, "data": {
                    "token": token.key
                }})
            else:
                return Response({"status": 400})
        else:
            return Response({"status": 400})

    def post(self, request):
        if request.user.username == "admin":
            user = reg_user(request)
            token = Token.objects.create(user=user)
            return Response({"status": 200, "data": {
                "token": token.key
            }})
        else:
            return Response({"status": 400})


class Operation(APIView):
    permission_classes = [permissions.IsAuthenticated, ]
    authentication_classes = (authentication.TokenAuthentication,)

    def post(self, request):
        trans_data = dict(request.data)
        trans_data = prepare_data(trans_data)
        is_fraud = predict_transaction(trans_data)
        trans_data["identifier"] = request.POST.get("identifier")
        transaction = TransactionSerializer(data=trans_data)
        if transaction.is_valid():
            transaction.save(ownerBank=request.user, prediction=is_fraud)
            return Response({"status": 200, "data": {
                "result": is_fraud
            }})
        else:
            return Response({"status": 400})

    def put(self, request):
        try:
            print(request.user.id)
            transaction = Transaction.objects.get(ownerBank=request.user.id, identifier=request.POST.get("identifier"))
            right_result = True if request.POST.get("result") == "true" else False
            transaction.prediction = right_result
            transaction.save()
            return Response({"status": 200})
        except Transaction.DoesNotExist:
            return Response({"status": 404})


class MLModel(APIView):
    permission_classes = [permissions.IsAuthenticated, ]
    authentication_classes = (authentication.TokenAuthentication,)

    def post(self, request):
        if request.user.username == "admin":
            Transaction.objects.all().delete()
            retrain_model()
            return Response({"status": 200})
        else:
            return Response({"status": 400})
