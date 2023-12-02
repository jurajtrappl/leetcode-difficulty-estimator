import json
import leetcode
import leetcode.auth

# get the next two values from your browser cookies
leetcode_session = "***REMOVED_LEETCODE_SESSION***"
csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)

configuration = leetcode.Configuration()

configuration.api_key["x-csrftoken"] = csrf_token
configuration.api_key["csrftoken"] = csrf_token
configuration.api_key["LEETCODE_SESSION"] = leetcode_session
configuration.api_key["Referer"] = "https://leetcode.com"
configuration.debug = False

api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

# Lets find out the list of names of all problems.
variables = {
    "categorySlug": "",
    "limit": 2950, # number of problems i found somewhere that is on leetcode in total
    "skip": 0,
    "filters": {}
}

graphql_request_problems_list = leetcode.GraphqlQuery(
    query="""
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
            problemsetQuestionList: questionList(categorySlug: $categorySlug, limit: $limit, skip: $skip, filters: $filters) {
                total: totalNum
                questions: data {
                    acRate
                    difficulty
                    title
                    titleSlug
                    topicTags {
                        name
                        id
                        slug
                    }
                    hasSolution
                }
            }
        }       
    """,
    variables=variables,
    operation_name="problemsetQuestionList"
)

problems_question_list = api_instance.graphql_post(body=graphql_request_problems_list).to_dict()["data"]["problemset_question_list"]
problems = [problem["title_slug"] for problem in problems_question_list["questions"]]

# For each problem, query the difficulty and description.
result = {}
for title_slug in problems:
    variables = {
        "titleSlug": title_slug
    }
    
    graphql_request = leetcode.GraphqlQuery(
        query="""
            query questionData($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    content
                    difficulty
                }
            }
        """,
        variables=variables,
        operation_name="questionData",
    )

    api_response = api_instance.graphql_post(body=graphql_request).to_dict()["data"]["question"]
    result[title_slug] = {
        "difficulty": api_response["difficulty"],
        "content": api_response["content"]
    }
    
# Write out results to file.
with open("leetcode_problems_dataset.json", "w") as f:
    json.dump(result, f)