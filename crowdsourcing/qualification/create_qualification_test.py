import boto.mturk.connection as mt
import boto.mturk.question as mt_q

ACCESS_ID = ''
SECRET_KEY = ''
HOST = 'mechanicalturk.sandbox.amazonaws.com'
# HOST = 'mechanicalturk.amazonaws.com'


def main():

    # Connect to MTurk
    connection = mt.MTurkConnection(aws_access_key_id=ACCESS_ID, aws_secret_access_key=SECRET_KEY, host=HOST)

    # Overview
    overview = mt_q.Overview()

    with open('instructions.txt') as f_in:
        instructions = f_in.read()

    overview.append_field('FormattedContent', '<![CDATA[%s]]>' % instructions)
    overview.append_field('Title', 'Answer the following questions:')

    # Question form
    question_form = mt_q.QuestionForm()
    question_form.append(overview)

    with open('questions.txt') as f_in:
        questions = [line.strip() for line in f_in]

    answers = [('Reference', 'reference'), ('Year', 'year'), ('Age', 'age'), ('People', 'people'), ('None', 'none')]

    for i, question in enumerate(questions):

        qc = mt_q.QuestionContent()
        qc.append_field('FormattedContent', '<![CDATA[%s]]>' % question)
        fta = mt_q.SelectionAnswer(min=1, max=1, style='radiobutton', selections=answers, type='text', other=False)
        q = mt_q.Question(identifier='q%d' % (i + 1), content=qc, answer_spec=mt_q.AnswerSpecification(fta), is_required=True)
        question_form.append(q)

    # Build the answer key
    with open('answer_key.xml') as f_in:
        answer_key = f_in.read()

    # Create the qualification type
    connection.create_qualification_type(name='What\'s Missing',
                                         description='Choose the best description of numbers inside texts',
                                         status='Active', test=question_form, answer_key=answer_key,
                                         test_duration=3600, retry_delay=600)


if __name__ == '__main__':
    main()
