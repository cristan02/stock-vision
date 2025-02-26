"use client";

import React from "react";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import Portfolio from "@/components/Portfolio";

export default function PortfolioCarosel({ portfolios }: any) {
  const responsive = {
    superLargeDesktop: {
      // the naming can be any, depends on you.
      breakpoint: { max: 5000, min: 3001 },
      items: 1,
    },
    largeDesktop: {
      breakpoint: { max: 3000, min: 1920 },
      items: 1,
    },
    desktop: {
      breakpoint: { max: 1920, min: 1024 },
      items: 1,
    },
    tablet: {
      breakpoint: { max: 1024, min: 501 },
      items: 1,
    },
    mobile: {
      breakpoint: { max: 520, min: 0 },
      items: 1,
    },
  };

  return (
    <div className="w-full min-h-56 h-full rounded md:col-span-3 ">
      <Carousel
        additionalTransfrom={0}
        arrows={false}
        autoPlay
        autoPlaySpeed={3000}
        centerMode={false}
        draggable={true}
        focusOnSelect
        infinite
        itemClass=" "
        keyBoardControl
        minimumTouchDrag={80}
        pauseOnHover
        renderArrowsWhenDisabled={false}
        // renderButtonGroupOutside
        // renderDotsOutside
        responsive={responsive}
        // rewind={false}
        // rewindWithAnimation={false}
        removeArrowOnDeviceType={["mobile"]}
        rtl={false}
        // shouldResetAutoplay
        showDots
        sliderClass=" "
        slidesToSlide={1}
        swipeable={true}
        ssr={true}
        className=" w-full"
      >
        {portfolios
          .map((item: any, index: any) => <Portfolio key={index} item={item} />)
          .slice(0, 2)}
      </Carousel>
    </div>
  );
}
