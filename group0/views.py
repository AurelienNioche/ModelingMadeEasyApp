from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader
import json


# def index(request):
#     return HttpResponse("Hello, world. You're at the group0 index.")


def index(request):
    # latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('group0/index.html')
    context = {
        # 'latest_question_list': latest_question_list,
    }
    return HttpResponse(template.render(context, request))