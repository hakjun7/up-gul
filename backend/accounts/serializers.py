from rest_framework import serializers
from allauth.account.adapter import get_adapter
from allauth.account.utils import setup_user_email
from rest_auth.registration.serializers import RegisterSerializer


class CustomRegisterSerializer(RegisterSerializer):
    age = serializers.IntegerField(required=False)
    region = serializers.CharField(required=False, max_length=100)
    mbti = serializers.CharField(required=False, max_length=10)

    def get_cleaned_data(self):
        data_dict = super().get_cleaned_data()
        data_dict['age'] = self.validated_data.get('age', '')
        data_dict['region'] = self.validated_data.get('region', '')
        data_dict['mbti'] = self.validated_data.get('mbti', '')
        return data_dict
