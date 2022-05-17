## 1. x변수에 26부터 1까지 역순으로 구성된 벡터를 저장하는 코드를 제시하세요.
x <- 26:1
x

## 2. y변수에 대문자 Z부터 A까지 역순으로 구성된 벡터를 저장하는 코드를 제시하세요.
y <- LETTERS[length(LETTERS):1]
y

## 3. 1번 문항의 x를 value로, 2번 문항의 y를 key로 하는 사전(dictionary)을 만들고 다음의 출력 결과가 나오는 코드를 제시하세요.
### 출력결과
### > x['b']
### b
### 2
names(x) <- y
x['B']
x

## 4. 1번 문항의 x변수를 역순으로 10개의 값을 출력할 수 있는 코드를 제시하세요.
x[length(x):(length(x)-9)]

## 5. 2번 문항의 y변수의 데이터 형태를 factor로 변경하고 데이터 형태를 확인할 수 있는 코드를 제시하세요.
y <- as.factor(y)
class(y)

## 6. 제곱미터를 입력값으로 하여 평형을 계산할 수 있는 my_house 함수를 작성해 다음의 예시 코드가 작동되도록 하고 그 코드 및 결과를 제시하세요. 참고) 1제곱미터 = 0.3025평.
### 예시코드
### my_house(runif(1,59,200))
my_house <- function(x){
  x*0.3025
}

my_house(runif(1,59,200))
my_house(2)
## 7. 동전을 n번 던져 앞면이 나온 횟수를 제시하는 my_coin함수를 작성해 다음의 예시코드가 작동되도록 하고 그 코드 및 결과를 제시하세요. 참고) rbinom() 함수 사용. 0은 뒷면, 1은 앞면.
### 예시코드
### my_coin(100)

my_coin <- function(x){
  sum(rbinom(x,1,prob=0.5))
}
my_coin(100)

## 8. movie데이터에서 country, imdb_score, duration, title_year열만 선택해 movie변수에 저장한 뒤 결측치를 확인할 수 있는 그래프를 제시하고 결과를 해석하세요.
movie <- read.csv('movie.csv',stringsAsFactors = FALSE)[c('country','imdb_score','duration','title_year')]
require(Hmisc)
md.pattern(movie,rotate.names=FALSE)
str(movie)

## 9. 8번 문항의 movie변수에서 결측치가 있는 레코드를 제거해 movie변수에 저장하는 코드를 제시하세요.
movie <- movie[complete.cases(movie),]
str(movie)

## 10. 9번 문항의 movie 변수에서 다음의 조건을 모두 만족하는 레코드를 movie_filtered 변수에 저장하는 코드를 제시하고 해당되는 레코드가 몇 개가 있는지를 제시하세요.
### 조건
### 1. 국가(country)가 USA 또는 UK
### 2. 출시년도(title_year)가 2003년 이전 또는 2010년 이후
### 3. 평점(imdb_score)가 8점 이상
### 4. 상영시간이 120분 이하
f1 <- movie$country == 'USA' | movie$country == 'UK'
f2 <- movie$title_year < 2003 | movie$title_year > 2010
f3 <- movie$imdb_score >= 8
f4 <- movie$duration <= 120

movie_filtered <- movie[f1&f2&f3&f4,]
str(movie_filtered)
movie_filtered

#### flights.csv 파일을 이용해 다음의 문항을 순서대로 해결하세요. ####

## 11. 각 항공사(AIRLINE)별 월(MONTH)별 평균 운항거리(DIST)를 구할 수 있는 코드를 제시하세요.
flights <- read.csv('flights.csv',stringsAsFactors = FALSE)
aggregate(DIST~AIRLINE+MONTH,flights,mean)

## 12. 요일(WEEKDAY)별 도착지연(ARR_DELAY)이 아닌 경우(0과 음수)의 평균 운항 시간(AIR_TIME)을 구할 수 있는 코드를 제시하세요.
aggregate(AIR_TIME~WEEKDAY,flights[flights$ARR_DELAY<=0,],mean)

## 13. flights 데이터에서 도착지연시간(ARR_DELAY)와 출발지연시간(DEP_DELAY)의 이상치를 파악할 수 있는 boxplot을 그리고 boxplot에 나타난 이상치들을 NA로 변경해 flights 변수에 저장하는 코드를 제시하세요.
box <- boxplot(flights$ARR_DELAY,flights$DEP_DELAY)
box$stats
flights$ARR_DELAY[flights$ARR_DELAY<box$stats[1,1]] <- NA
flights$ARR_DELAY[flights$ARR_DELAY>box$stats[5,1]] <- NA
flights$DEP_DELAY[flights$DEP_DELAY<box$stats[1,2]] <- NA
flights$DEP_DELAY[flights$DEP_DELAY>box$stats[5,2]] <- NA
flights <- flights[c('ARR_DELAY','DEP_DELAY')]
flights

## 14. 13번의 flights 변수에서 mice패키지를 사용해 결측치를 대체하는 코드를 제시하고 그 결과로 나오는 density plot을 제시하고 그래프를 해석하세요.
library(mice)
re_flights <- mice(flights,m=10,seed=123,print=TRUE)
densityplot(re_flights)
