from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template('pre_questionnaire/index.html')
    context = {}
    return HttpResponse(template.render(context, request))